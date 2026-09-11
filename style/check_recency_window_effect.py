"""
유저당 사용 경기 수(recency window)를 줄이면 K-means 군집 구조(실루엣)가 개선되는지
실험하는 진단 스크립트.

가설(대화 중 제기됨): "유저당 최근 최대 100경기까지 다 모아서 평균 내면, 특히
through_pass_ratio 같은 상황 의존적 feature가 평균으로 수렴해 개인차가 옅어지고,
그래서 실루엣이 0.3대에 머무는 것 아니냐."

이미 표준편차/IQR 비교로 through_pass_ratio 등 일부 feature에서 "경기 수가 많을수록
평균에 수렴하는" 약한 경향을 확인했다. 이 스크립트는 그 가설을 실제 재군집화 결과
(실루엣 점수)로 직접 검증한다.

**실험 설계**: population을 고정한다 — preprocess_style.py와 동일하게 "정상종료 경기
50건 이상"인 유저(현재 137명, train_style.py의 baseline과 동일 모집단)만 대상으로
하고, 그 안에서 유저별로 최근 N경기(N=30/50/전체)만 골라 비율 feature를 다시 계산한다.
모집단을 고정해야 "경기 수가 줄어서 유저 구성 자체가 달라진 효과"와 "같은 유저를
최근 N경기로만 보는 효과"가 섞이지 않는다.

2026-09-08: 슛 모델(in_penalty+heading 통합)이 헤딩축/슛위치축으로 분리되면서, 이
스크립트도 패스/헤딩/슛위치 3개 모델을 각각 검증하도록 갱신했다.

실행: ./venv/Scripts/python.exe style/check_recency_window_effect.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import preprocess_style as preprocess
import train_style as train

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "style")
WINDOW_SIZES = [30, 50, None]  # None = 전체 경기 (현재 baseline과 동일)
K_RANGE = range(2, 7)
RANDOM_SEED = 42
# ====================================================


def extract_rows_with_date(df):
    """preprocess.extract_match_style_rows와 동일한 필터/컬럼이지만, 최근 N경기를
    고를 수 있게 match_date를 같이 남긴다."""
    normal = df[df[preprocess.MATCH_END_TYPE_COL] == preprocess.NORMAL_MATCH_END_TYPE]
    through_pass_try = normal[preprocess.THROUGH_PASS_TRY_COLS].fillna(0).sum(axis=1)
    return pd.DataFrame({
        "ouid": normal[preprocess.OUID_COL].values,
        "match_date": pd.to_datetime(normal["matchDate"], errors="coerce").values,
        "pass_try": normal[preprocess.PASS_TRY_COL].fillna(0).values,
        "short_pass_try": normal[preprocess.SHORT_PASS_TRY_COL].fillna(0).values,
        "long_pass_try": normal[preprocess.LONG_PASS_TRY_COL].fillna(0).values,
        "through_pass_try": through_pass_try.values,
        "driven_ground_pass_try": normal[preprocess.DRIVEN_GROUND_PASS_TRY_COL].fillna(0).values,
        "shoot_total": normal[preprocess.SHOOT_TOTAL_COL].fillna(0).values,
        "shoot_heading": normal[preprocess.SHOOT_HEADING_COL].fillna(0).values,
        "shoot_out_penalty": normal[preprocess.SHOOT_OUT_PENALTY_COL].fillna(0).values,
    })


def _safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else np.nan


def aggregate_with_window(rows_df, qualifying_ouids, window):
    """qualifying_ouids(고정 모집단)만, 유저별 최근 window경기(None이면 전부)로 비율을 낸다."""
    records = []
    for ouid, g in rows_df.groupby("ouid"):
        if ouid not in qualifying_ouids:
            continue
        g = g.sort_values("match_date", ascending=False)
        if window is not None:
            g = g.head(window)
        pass_try = g["pass_try"].sum()
        shoot_total = g["shoot_total"].sum()
        records.append({
            "ouid": ouid,
            "n_matches": len(g),
            "short_pass_ratio": _safe_ratio(g["short_pass_try"].sum(), pass_try),
            "long_pass_ratio": _safe_ratio(g["long_pass_try"].sum(), pass_try),
            "through_pass_ratio": _safe_ratio(g["through_pass_try"].sum(), pass_try),
            "driven_ground_pass_ratio": _safe_ratio(g["driven_ground_pass_try"].sum(), pass_try),
            "heading_shoot_ratio": _safe_ratio(g["shoot_heading"].sum(), shoot_total),
            "out_penalty_shoot_ratio": _safe_ratio(g["shoot_out_penalty"].sum(), shoot_total),
        })
    feature_cols = (
        preprocess.PASS_FEATURE_COLUMNS + preprocess.HEADING_FEATURE_COLUMNS
        + preprocess.SHOT_LOCATION_FEATURE_COLUMNS
    )
    return pd.DataFrame(records).dropna(subset=feature_cols)


def best_silhouette(style_df, feature_columns, k_range=K_RANGE):
    x_scaled = StandardScaler().fit_transform(style_df[feature_columns].astype(float))
    best_k, best_score = None, -1.0
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(x_scaled)
        score = silhouette_score(x_scaled, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


def main():
    print("1) match_team_data.csv 로드 (matchDate 포함)")
    matches = preprocess.load_matches(preprocess.MATCHES_FILE)
    rows_df = extract_rows_with_date(matches)

    print("2) 고정 모집단 산출 (정상종료 경기 50건 이상 유저 - 현재 baseline과 동일)")
    counts = rows_df.groupby("ouid").size()
    qualifying_ouids = set(counts[counts >= preprocess.MIN_MATCHES_PER_USER].index)
    print(f"  모집단 유저 수: {len(qualifying_ouids)}")

    results = []
    for window in WINDOW_SIZES:
        label = f"최근 {window}경기" if window is not None else "전체(baseline)"
        print(f"\n3) window={label} 로 재집계 및 k=2~6 탐색")
        style_df = aggregate_with_window(rows_df, qualifying_ouids, window)
        print(f"  실제 유저 수: {len(style_df)} (분모 0으로 제외된 유저 있으면 모집단보다 적을 수 있음)")

        row = {"window": label, "n_users": len(style_df)}
        for key, feature_columns, _ in train.MODEL_SPECS:
            k, score = best_silhouette(style_df, feature_columns)
            print(f"  {key} 모델: k={k}, silhouette={score:.4f}")
            row[f"{key}_best_k"] = k
            row[f"{key}_silhouette"] = score
        results.append(row)

    result_df = pd.DataFrame(results)
    print("\n=== 최종 비교 ===")
    print(result_df.round(4).to_string(index=False))

    out_path = os.path.join(OUTPUT_DIR, "recency_window_effect.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== 유저당 사용 경기 수(recency window)별 실루엣 비교 ===\n")
        f.write("모집단 고정(정상종료 50경기 이상 유저), 유저별 최근 N경기만으로 비율 feature 재계산.\n\n")
        f.write(result_df.round(4).to_string(index=False))
    print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    main()
