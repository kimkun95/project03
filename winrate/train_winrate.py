"""
승부예측(승/패 이진분류) baseline 모델 학습 실행.

데이터 파싱/feature 조립은 preprocess_winrate.py, 저장은 utils.py에 맡기고, 여기서는
train/validation/test 분할과 모델 학습만 담당한다.

feature는 포지션 그룹별 평균 능력치 4개(attack_avg_score, mid_avg_score, defense_avg_score,
gk_avg_score) + tier다. 포메이션/팀컬러/style_fit_score는 아직 미해결(자리 조합표 미작성,
API 응답 존재 미확인, USER_STYLE_PROFILE 미구축)이라 이번 버전에서는 제외했다. 나중에 이
feature들이 추가된 버전과 "같은 데이터, 같은 split"으로 비교할 예정이므로, split에 쓰는
random seed를 고정하고 어떤 match_id/ouid가 train/validation/test 중 어디로 갔는지
data/winrate/split_assignment.json에 저장해둔다.

실사용/저장하는 모델은 sklearn Pipeline(StandardScaler + LogisticRegression)이다.
tier(2200~2400대)와 avg_score 4개(50~100대)는 값의 스케일 차이가 커서, 정규화가 걸리는
LogisticRegression에 그대로 넣으면 tier가 부당하게 억눌리기 때문에 StandardScaler를 앞에 둔다.

test셋은 "최종 선택 후 딱 한 번만" 보는 용도다. feature/모델을 바꿔가며 반복 실행할 때는
validation만 보고 판단해야 하는데, 예전 버전은 매번 test까지 같이 계산/출력해서 실제로는
여러 번 들여다보는 실수를 했었다(=validation처럼 써버려서 최종 숫자의 신뢰도가 깎임).
이를 막기 위해 EVALUATE_TEST_SET 플래그를 뒀다 — 기본값 False에서는 validation까지만
계산하고, "이제 진짜 최종 확인"이라고 확신할 때만 True로 바꿔서 실행한다.

sklearn LogisticRegression은 계수의 p-value를 주지 않는다. 계수 유의성을 확인할 수 있도록
같은 train 데이터로 statsmodels.Logit을 한 번 더 학습해 리포트(summary 텍스트)에만 사용한다.
이 statsmodels 모델은 저장/서빙에는 쓰지 않는다. 두 모델의 계수는 스케일링 여부가 달라
크기가 다를 수 있다 (sklearn=표준화 스케일, statsmodels=원 스케일) — summary에 이 점을 명시한다.
"""

import os

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

import preprocess_winrate as preprocess
import utils

# ============ CONFIG ============
# 실행 위치(cwd)에 관계없이 항상 저장소 루트 기준 경로를 쓰도록 스크립트 파일 위치에서 계산한다.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "winrate")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
RANDOM_SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# 포메이션/팀컬러/style_fit_score가 나중에 추가될 것을 감안해 feature 목록을 한 곳에서
# 관리한다 (preprocess_winrate.GROUP_SCORE_COLUMNS와 순서 일치).
FEATURE_COLUMNS = preprocess.GROUP_SCORE_COLUMNS + ["tier"]

# feature/모델을 실험하는 동안에는 반드시 False로 둔다 — validation만 보고 판단하고,
# "이제 이 버전으로 확정한다"고 결론 낸 뒤 딱 한 번 True로 바꿔서 최종 test 성능만 확인한다.
# 여러 번 실행해보며 test 결과를 보고 feature를 고르면, test가 사실상 validation처럼
# 쓰이게 돼 최종 숫자가 실제보다 낙관적으로 부풀려진다 (test set leakage).
EVALUATE_TEST_SET = False
# ====================================================


def split_dataset(feature_df, output_dir):
    """train/validation/test로 랜덤 분할(기본 60/20/20)하고, 나중 비교를 위해 어떤
    match_id/ouid가 어디로 갔는지 output/split_assignment.json에 저장한다.

    validation은 이번 baseline 자체 평가에는 쓰지 않지만, 이후 style_fit_score를 추가한
    버전과 "test셋은 절대 건드리지 않고" 모델을 비교/선택할 때 쓰기 위해 미리 분리해둔다.
    """
    train_val_df, test_df = train_test_split(
        feature_df, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )
    val_ratio_within_train_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio_within_train_val, random_state=RANDOM_SEED
    )

    assignment = (
        [{"match_id": r.match_id, "ouid": r.ouid, "split": "train"} for r in train_df.itertuples()]
        + [{"match_id": r.match_id, "ouid": r.ouid, "split": "val"} for r in val_df.itertuples()]
        + [{"match_id": r.match_id, "ouid": r.ouid, "split": "test"} for r in test_df.itertuples()]
    )
    utils.save_json({
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "rows": assignment,
    }, f"{output_dir}/split_assignment.json")

    print(
        f"  [split] train {len(train_df)}건 / val {len(val_df)}건 / test {len(test_df)}건 "
        f"-> {output_dir}/split_assignment.json"
    )
    return train_df, val_df, test_df


def build_pipeline():
    """StandardScaler + LogisticRegression으로 이뤄진 sklearn Pipeline을 만든다."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ])


def evaluate_pipeline(pipeline, df, feature_columns):
    """주어진 split(val 또는 test)에 대해 accuracy와 50% 대비 유의성(binomial test)을 계산한다."""
    X_eval = df[feature_columns].astype(float)
    y_eval = df["result"].astype(int)
    predicted = pipeline.predict(X_eval)

    correct = int((predicted == y_eval).sum())
    n = len(y_eval)
    accuracy = correct / n if n else float("nan")

    sig_test = binomtest(correct, n, p=0.5) if n else None
    p_value = sig_test.pvalue if sig_test else float("nan")

    return accuracy, p_value, correct, n


def fit_statsmodels_report(train_df, feature_columns):
    """계수의 p-value를 확인하기 위해 원 스케일 feature로 statsmodels Logit을 학습한다.

    이 모델은 저장/서빙에는 쓰지 않고 summary 텍스트 리포트 전용이다.
    """
    X_train = sm.add_constant(train_df[feature_columns].astype(float), has_constant="add")
    y_train = train_df["result"].astype(int)
    return sm.Logit(y_train, X_train).fit(disp=0)


def build_summary_text(pipeline, statsmodels_result, feature_columns, train_size, match_rate,
                        val_metrics, test_metrics=None):
    """test_metrics는 EVALUATE_TEST_SET=True로 최종 확인할 때만 넘긴다. None이면 summary에
    test 관련 줄을 아예 넣지 않는다 — 값을 채워 넣지 않고 "아직 안 봤다"는 상태 자체를
    남겨서, 실수로 test 숫자가 리포트에 새어 들어가는 걸 막는다."""
    val_accuracy, val_p_value, val_correct, n_val = val_metrics
    if test_metrics is not None:
        test_accuracy, test_p_value, test_correct, n_test = test_metrics

    clf = pipeline.named_steps["clf"]
    sklearn_coef_lines = [
        f"  {name}: {coef:.4f} (표준화 스케일)"
        for name, coef in zip(feature_columns, clf.coef_[0])
    ]

    lines = [
        "=== 승부예측 Baseline 모델 요약 ===",
        f"feature: {feature_columns}",
        "",
        "[sklearn Pipeline 계수] (StandardScaler로 표준화된 스케일 - 저장/서빙에 쓰는 모델)",
        f"  intercept: {clf.intercept_[0]:.4f}",
        *sklearn_coef_lines,
        "",
        "[statsmodels 계수/p-value] (원 스케일 - 리포트 전용, 저장/서빙에는 안 씀)",
        str(statsmodels_result.summary()),
        "",
        f"train 표본 수: {train_size}",
        f"validation 표본 수: {n_val} (정답 {val_correct}건)",
        f"validation accuracy: {val_accuracy:.4f}",
        f"validation 50% 우연 대비 이항검정 p-value: {val_p_value:.4f}",
        (
            f"test 표본 수: {n_test} (정답 {test_correct}건)\n"
            f"test accuracy: {test_accuracy:.4f}\n"
            f"test 50% 우연 대비 이항검정 p-value: {test_p_value:.4f}\n"
            "  -> p < 0.05이면 우연(50%)보다 유의미하게 낫다고 판단 가능."
            if test_metrics is not None
            else "test: 아직 평가 안 함 (EVALUATE_TEST_SET=False — feature/모델 확정 전) "
            "— validation만 보고 판단할 것."
        ),
        f"player_stats_final.csv spId 매칭률: {match_rate:.1f}%",
        "",
        "[참고] sklearn 계수는 표준화된 스케일, statsmodels 계수는 원 스케일이라 크기가 "
        "서로 다를 수 있다. 방향(부호)이 같은지만 참고하고, 유의성 판단은 statsmodels "
        "p-value를 쓴다.",
        "[참고] feature/모델을 확정하기 전까지는 이 validation 결과만 보고 판단한다. "
        "test셋은 EVALUATE_TEST_SET=True로 바꿔 최종 확정 후 딱 한 번만 확인한다 — 여러 "
        "버전을 시도하며 test까지 매번 보면 test가 사실상 validation처럼 쓰여 최종 수치가 "
        "부풀려진다(test set leakage).",
        "[한계] player_stats_final.csv의 스탯 수치가 강화단계(spGrade)를 반영한 값인지 "
        "기본(0강) 값인지 확인되지 않았다. 확인 전까지는 강화단계를 무시하고 spId만으로 "
        "매칭한 근사치로 취급한다.",
        "[한계] 유저 실력(손가락) 차이는 feature로 통제하지 못하며, tier로 부분적으로만 "
        "간접 반영된다.",
    ]
    return "\n".join(line for line in lines if line != "")


def main():
    preprocess.assert_no_leakage(FEATURE_COLUMNS)

    print("1) matches.jsonl 로드 및 행 추출")
    matches = preprocess.load_matches(preprocess.MATCHES_FILE)
    rows_df = preprocess.extract_rows(matches)

    print("2) player_stats_final.csv 로드 및 포지션 그룹별 avg_score 계산")
    player_card_df = preprocess.load_player_cards(preprocess.PLAYER_CARD_FILE)
    feature_df, match_rate = preprocess.assemble_feature_table(rows_df, player_card_df)

    print("3) train/validation/test 분할")
    train_df, val_df, test_df = split_dataset(feature_df, OUTPUT_DIR)

    print("4) sklearn Pipeline 학습 및 평가")
    pipeline = build_pipeline()
    pipeline.fit(train_df[FEATURE_COLUMNS].astype(float), train_df["result"].astype(int))
    val_metrics = evaluate_pipeline(pipeline, val_df, FEATURE_COLUMNS)
    print(f"  validation accuracy: {val_metrics[0]:.4f} (p-value vs 50%: {val_metrics[1]:.4f})")
    if EVALUATE_TEST_SET:
        test_metrics = evaluate_pipeline(pipeline, test_df, FEATURE_COLUMNS)
        print(f"  test accuracy: {test_metrics[0]:.4f} (p-value vs 50%: {test_metrics[1]:.4f})")
    else:
        test_metrics = None
        print("  test: 평가 생략 (EVALUATE_TEST_SET=False) - feature/모델 확정 전에는 "
              "validation만 보고 판단할 것.")

    print("5) statsmodels로 계수 p-value 리포트 생성")
    statsmodels_result = fit_statsmodels_report(train_df, FEATURE_COLUMNS)

    print("6) 결과 저장")
    summary_text = build_summary_text(
        pipeline, statsmodels_result, FEATURE_COLUMNS, len(train_df), match_rate,
        val_metrics, test_metrics,
    )
    utils.save_model(pipeline, f"{MODEL_DIR}/win_prediction_baseline_pipeline.joblib")
    utils.save_text(summary_text, f"{OUTPUT_DIR}/win_prediction_baseline_summary.txt")
    print(f"  [저장] 모델 -> {MODEL_DIR}/win_prediction_baseline_pipeline.joblib")
    print(f"  [저장] 요약 -> {OUTPUT_DIR}/win_prediction_baseline_summary.txt")


if __name__ == "__main__":
    main()
