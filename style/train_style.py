"""
플레이스타일 진단(K-means 군집화, 비지도학습) baseline 실행.

데이터 파싱/feature 조립은 preprocess_style.py, 저장은 utils.py에 맡기고, 여기서는
표준화 -> k 탐색 -> 최종 군집화 -> 결과 저장만 담당한다.

CLAUDE.md 원칙 #4("규칙 기반으로 미리 유형 개수를 정하지 않는다")를 그대로 따른다:
k=2~6을 전부 시도해 실루엣 점수가 가장 높은 k를 사후에 고르고, 각 군집에 사람이 붙일
이름(예: "스루패스 위주形")은 이 스크립트가 정하지 않는다 — cluster_summary에 군집별
feature 평균만 남겨서, 그 숫자를 보고 팀이 나중에 라벨을 붙인다.

2026-09-07 결정: "패스+슛 6개 feature 통합 1개 모델" 대신 **패스 모델 + 슛 모델을
독립적으로 군집화**한다 (preprocess_style.py docstring에 실험 근거 정리됨).

2026-09-08 결정: 슛 모델(in_penalty+heading 2개 feature 통합)도 같은 이유로 다시
쪼갰다 — **헤딩축(heading_shoot_ratio 단독) + 슛위치축(out_penalty_shoot_ratio 단독)**
2개로 분리. 그래서 유저당 군집 라벨이 `pass_style_type`, `heading_style_type`,
`shot_location_style_type` 3개로 나온다. K-means 외 GMM 등도 비교해봤지만 최종적으로
K-means만 쓰기로 팀이 결정했다.

PCA는 군집화에 쓰지 않고, 모델별 시각화용 좌표를 뽑는 용도로만 별도로 돌린다. 헤딩축/
슛위치축처럼 feature가 1개뿐인 모델은 PCA 좌표도 1차원(pca_x만)만 나온다.
"""

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import preprocess_style as preprocess
import utils

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "style")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
RANDOM_SEED = 42
K_RANGE = range(2, 7)  # CLAUDE.md 원칙: k=2~6을 전부 시도

# (모델 키, feature 컬럼 목록, 유저별 군집 라벨 컬럼명) — 3개 모델을 이 목록만으로 루프 처리한다.
MODEL_SPECS = [
    ("pass", preprocess.PASS_FEATURE_COLUMNS, "pass_style_type"),
    ("heading", preprocess.HEADING_FEATURE_COLUMNS, "heading_style_type"),
    ("shot_location", preprocess.SHOT_LOCATION_FEATURE_COLUMNS, "shot_location_style_type"),
]
# ====================================================


def select_best_k(x_scaled, k_range=K_RANGE):
    """k별로 KMeans를 돌려 실루엣 점수를 계산하고, 가장 높은 k를 고른다.

    반환값: (best_k, {k: (labels, silhouette_score)} 전체 결과)
    """
    results = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(x_scaled)
        score = silhouette_score(x_scaled, labels)
        results[k] = (labels, score, kmeans)

    best_k = max(results, key=lambda k: results[k][1])
    return best_k, results


def run_style_clustering(style_df, feature_columns):
    """표준화 -> k=2~6 탐색 -> 최적 k 선택 -> 시각화용 PCA 좌표까지 한 번에 계산한다.

    패스 모델(4개 feature), 헤딩축/슛위치축(각 1개 feature)에 똑같이 재사용한다.
    feature가 1개뿐이면 PCA 좌표도 1차원만 나온다(n_components=1).
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(style_df[feature_columns].astype(float))

    best_k, k_search_results = select_best_k(x_scaled)
    labels, best_score, best_kmeans = k_search_results[best_k]

    n_components = min(2, len(feature_columns))
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    coords = pca.fit_transform(x_scaled)

    return {
        "labels": labels,
        "best_k": best_k,
        "best_score": best_score,
        "k_search_results": k_search_results,
        "scaler": scaler,
        "kmeans": best_kmeans,
        "pca": pca,
        "coords": coords,
    }


def build_cluster_summary(style_df, feature_columns, label_column):
    """군집별 표본 수와 feature 평균을 낸다 — 사후 라벨링(이름 붙이기)에 쓰는 참고 표."""
    summary = style_df.groupby(label_column)[feature_columns].mean()
    summary["n_users"] = style_df.groupby(label_column).size()
    return summary.reset_index()


def build_summary_text(n_users, min_matches, results, summaries):
    """results/summaries: MODEL_SPECS 순서와 동일한 리스트 (모델별 run_style_clustering
    결과, build_cluster_summary 결과)."""
    lines = [
        "=== 플레이스타일 진단(K-means) Baseline 요약 ===",
        f"유저 최소 표본 경기 수(MIN_MATCHES_PER_USER): {min_matches}",
        f"군집화 대상 유저 수: {n_users}",
        "",
        "[설계 결정] 패스/헤딩/슛위치 3개 모델을 독립적으로 군집화한다. 6개 feature를 한",
        "공간에서 통합 군집화하면 실루엣 0.23~0.25, 패스+슛(2개 feature) 통합도 0.36에",
        "그쳤는데, 헤딩축/슛위치축까지 마저 쪼개니 각각 0.59/0.56까지 개선됐다.",
        "dribble_intensity/possession/block_ratio 등도 시도했지만 전부 실루엣을 낮춰",
        "(실력·경기 흐름과 얽혀 있어 스타일 신호를 희석시킴) 최종 feature에서 뺐다",
        "(preprocess_style.py docstring 참고).",
    ]

    for (key, feature_columns, label_column), result, summary in zip(MODEL_SPECS, results, summaries):
        lines.append("")
        lines.append(f"--- {key} 모델 ---")
        lines.append(f"feature: {feature_columns}")
        for k in sorted(result["k_search_results"]):
            _, score, _ = result["k_search_results"][k]
            marker = "  <- 선택" if k == result["best_k"] else ""
            lines.append(f"  k={k}: silhouette={score:.4f}{marker}")
        lines.append(f"[선택된 k] {result['best_k']}")
        lines.append("[군집별 feature 평균]")
        lines.append(summary.to_string(index=False))

    lines.append("")
    lines.append("[한계] 최소 표본 경기 수 하한선(MIN_MATCHES_PER_USER)이 팀 논의로 "
                  "확정되지 않아 임시값을 쓰고 있다 (preprocess_style.py 참고).")
    lines.append("[한계] 실루엣 점수가 낮은 모델(패스)은 유저의 플레이 성향이 몇 개 뚜렷한 "
                  "그룹으로 딱 떨어지기보다 연속적인 스펙트럼에 가깝다는 뜻으로 해석하며, "
                  "완전히 분리된 유형이 아니라 '가장 가까운 성향' 정도의 관찰적 진단으로 "
                  "제시한다.")
    lines.append("[한계] long_pass_ratio/through_pass_ratio가 division(티어)과 상관관계가 "
                  "있음을 확인했다 — 관찰적 진단이라는 설계 선택에 따른 의도적 트레이드오프이며 "
                  "정규화로 보정하지 않는다 (style/README.md 참고).")
    lines.append("[참고] 헤딩(방식)과 박스 안 슈팅(위치)은 반대 개념이 아니다 — 헤딩은 대부분 "
                  "박스 안 가까운 거리에서 나오므로 두 값이 자연히 같이 높게 나올 수 있다. "
                  "헤딩축은 '방식'(헤딩 vs 발), 슛위치축은 '위치'(박스 안 vs 중거리)라는 서로 "
                  "다른 축으로 해석해야 한다.")
    return "\n".join(lines)


def main():
    print("1) match_team_data.csv 로드 및 (match, ouid) 행 추출")
    matches = preprocess.load_matches(preprocess.MATCHES_FILE)
    rows_df = preprocess.extract_match_style_rows(matches)

    print("2) 유저 단위 집계 및 패스/헤딩/슛위치 비율 feature 계산")
    style_df = preprocess.aggregate_user_style(rows_df)
    if len(style_df) < max(K_RANGE):
        raise ValueError(
            f"군집화 대상 유저가 {len(style_df)}명뿐이라 k 최대값({max(K_RANGE)})보다 "
            "적다. 유저 수를 늘리거나 K_RANGE를 줄여야 한다."
        )

    style_df = style_df.copy()
    results = []
    summaries = []
    model_dict = {}
    for i, (key, feature_columns, label_column) in enumerate(MODEL_SPECS, start=3):
        print(f"{i}) {key} 모델 군집화 (표준화 -> k=2~6 탐색 -> 최적 k 선택 -> PCA 좌표)")
        result = run_style_clustering(style_df, feature_columns)
        print(f"  선택된 k: {result['best_k']} (silhouette={result['best_score']:.4f})")

        style_df[label_column] = result["labels"]
        style_df[f"{key}_pca_x"] = result["coords"][:, 0]
        if result["coords"].shape[1] > 1:
            style_df[f"{key}_pca_y"] = result["coords"][:, 1]

        results.append(result)
        summaries.append(build_cluster_summary(style_df, feature_columns, label_column))
        model_dict[key] = {"scaler": result["scaler"], "kmeans": result["kmeans"], "pca": result["pca"]}

    print(f"{len(MODEL_SPECS) + 3}) 결과 저장")
    summary_text = build_summary_text(len(style_df), preprocess.MIN_MATCHES_PER_USER, results, summaries)

    utils.save_model(model_dict, os.path.join(MODEL_DIR, "style_diagnosis_baseline.joblib"))
    utils.ensure_dir(OUTPUT_DIR)
    style_df.to_csv(os.path.join(OUTPUT_DIR, "user_style_profile.csv"), index=False)
    utils.save_text(summary_text, os.path.join(OUTPUT_DIR, "style_diagnosis_baseline_summary.txt"))

    print(f"  [저장] 모델(모델별 scaler+kmeans+pca) -> {MODEL_DIR}/style_diagnosis_baseline.joblib")
    print(f"  [저장] 유저별 스타일 프로필 -> {OUTPUT_DIR}/user_style_profile.csv")
    print(f"  [저장] 요약 -> {OUTPUT_DIR}/style_diagnosis_baseline_summary.txt")


if __name__ == "__main__":
    main()
