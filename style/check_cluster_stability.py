"""
K-means 군집 결과가 random_state(초기값)에 따라 흔들리지 않는지 검증하는 안정성 진단.

train_style.py는 k=2~6마다 random_state=42로 딱 1번씩만 KMeans를 돌려 실루엣이 가장
높은 k를 고른다. 실루엣은 "군집 간 분리가 얼마나 뚜렷한가"만 잴 뿐, "그 k로 나눴을 때
매번 같은 사람들이 같은 군집으로 묶이는가"(재현성)는 알려주지 않는다. 이 스크립트는
같은 k에 대해 random_state를 여러 번 바꿔가며 반복 실행하고, 실행 결과끼리
Adjusted Rand Index(ARI)로 군집 배정이 얼마나 일치하는지 재서 그 재현성을 확인한다.

- 실루엣 점수: 군집 간 분리도(구조가 있는지) — train_style.py가 이미 확인함
- ARI: 재현성/안정성(같은 구조를 매번 찾아내는지) — 이 스크립트가 추가로 확인하는 것

원본 데이터를 그대로 다시 로드해 preprocess_style.py와 완전히 같은 방식으로 집계하므로,
train_style.py의 결과와 독립적으로 재현 가능하다. train_style.py의 MODEL_SPECS(패스/
헤딩/슛위치 3개 모델) 정의를 그대로 재사용해 모델이 늘거나 줄어도 이 스크립트를 따로
고칠 필요가 없게 했다(2026-09-08, 슛 모델이 헤딩축/슛위치축으로 분리되면서 갱신).

실행: ./venv/Scripts/python.exe style/check_cluster_stability.py
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import preprocess_style as preprocess
import train_style as train
import utils
from train_style import K_RANGE

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "style")
N_RUNS = 50  # 시드(random_state)를 바꿔가며 반복 실행할 횟수
ARI_STABLE_THRESHOLD = 0.9  # 이 값 이상이면 "안정적"으로 판단
ARI_WEAK_THRESHOLD = 0.7  # 이 값 미만이면 "재현성 약함"으로 판단
# ====================================================


def run_stability_check(x_scaled, k_range=K_RANGE, n_runs=N_RUNS):
    """k별로 random_state 0~n_runs-1로 KMeans를 반복 실행해 실루엣/ARI 분포를 낸다.

    반환: k, silhouette_mean/std, ari_mean/min, ari_stable_pct(0.9 이상 비율)을 담은 DataFrame.
    """
    rows = []
    for k in k_range:
        silhouettes = []
        label_runs = []
        for seed in range(n_runs):
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(x_scaled)
            silhouettes.append(silhouette_score(x_scaled, labels))
            label_runs.append(labels)

        aris = np.array([
            adjusted_rand_score(label_runs[i], label_runs[j])
            for i, j in combinations(range(n_runs), 2)
        ])
        silhouettes = np.array(silhouettes)
        rows.append({
            "k": k,
            "silhouette_mean": silhouettes.mean(),
            "silhouette_std": silhouettes.std(),
            "ari_mean": aris.mean(),
            "ari_min": aris.min(),
            "ari_stable_pct": (aris >= ARI_STABLE_THRESHOLD).mean() * 100,
        })
    return pd.DataFrame(rows)


def _judge(ari_mean):
    if ari_mean >= ARI_STABLE_THRESHOLD:
        return "안정적"
    if ari_mean >= ARI_WEAK_THRESHOLD:
        return "보통"
    return "재현성 약함"


def build_summary_text(model_results):
    """model_results: [(key, result_df, actual_k), ...] (train.MODEL_SPECS 순서)."""
    lines = [
        "=== 스타일 진단 K-means 안정성(재현성) 진단 ===",
        f"각 k에 대해 random_state 0~{N_RUNS - 1}로 KMeans를 {N_RUNS}번씩 반복 실행하고,",
        "실행 결과끼리 Adjusted Rand Index(ARI)로 군집 배정이 얼마나 일치하는지 측정했다.",
        "ARI=1.0은 완전히 동일한 배정, 0에 가까울수록 시드마다 배정이 달라짐(불안정)을 뜻한다.",
    ]

    for key, result, actual_k in model_results:
        lines.append("")
        lines.append(f"--- {key} 모델 ---")
        lines.append(result.round(4).to_string(index=False))
        lines.append(
            f"[train_style.py가 실제로 채택한 k] {actual_k} (random_state=42 단일 실행 기준) -> "
            f"{_judge(result.set_index('k').loc[actual_k, 'ari_mean'])}"
        )

    lines.append("")
    lines.append(
        f"[해석 기준] ari_mean >= {ARI_STABLE_THRESHOLD}: 안정적 (시드 무관 재현) / "
        f"{ARI_WEAK_THRESHOLD} 이상: 보통 / 미만: 재현성 약함(실루엣이 높아도 신중히 해석)."
    )
    lines.append(
        "[참고] 위 표의 k별 silhouette_mean은 이 스크립트가 시드 50개에 걸쳐 평균낸 값이라, "
        "train_style.py가 random_state=42 한 번만 돌려 채택한 k와 최고 평균 실루엣의 k가 "
        "다를 수 있다 (근소한 차이일 때 시드에 따라 순위가 뒤집힐 수 있음 — 그 자체가 이 "
        "진단의 요점이다). 그래서 비교 기준은 train_style.py가 실제로 저장한 k로 고정한다."
    )
    return "\n".join(lines)


def main():
    print("1) match_team_data.csv 로드 및 유저 단위 집계 (train_style.py와 동일 로직)")
    matches = preprocess.load_matches(preprocess.MATCHES_FILE)
    rows_df = preprocess.extract_match_style_rows(matches)
    style_df = preprocess.aggregate_user_style(rows_df)

    model_results = []
    for i, (key, feature_columns, _) in enumerate(train.MODEL_SPECS, start=2):
        print(f"\n{i}) {key} 모델 안정성 검증 (k={list(K_RANGE)}, 각 {N_RUNS}회 반복)")
        x_scaled = StandardScaler().fit_transform(style_df[feature_columns].astype(float))
        result = run_stability_check(x_scaled)
        print(result.round(4).to_string(index=False))

        # train_style.py와 완전히 같은 방식(random_state=42 단일 실행)으로 실제 채택 k를 구한다
        # (이 스크립트 자체의 50-시드 평균 실루엣 1위 k와는 다를 수 있음 — 요약문 참고).
        actual_k, _ = train.select_best_k(x_scaled)
        model_results.append((key, result, actual_k))

    summary_text = build_summary_text(model_results)
    utils.ensure_dir(OUTPUT_DIR)
    utils.save_text(summary_text, os.path.join(OUTPUT_DIR, "cluster_stability_summary.txt"))
    print(f"\n[저장] {OUTPUT_DIR}/cluster_stability_summary.txt")


if __name__ == "__main__":
    main()
