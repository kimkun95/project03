"""
K-means 군집 결과가 데이터 이상치(저활동/특정 티어 쏠림 등)로 인한 착시가 아닌지
확인하는 사후 검증 스크립트.

train_style.py가 저장한 data/style/user_style_profile.csv의 군집 라벨
(pass_style_type, heading_style_type, shot_location_style_type)을 원본 경기 데이터
(data/style/match_team_data.csv)와 다시 연결해, 군집 평균 feature 값만으로는 안 보이는
것들을 확인한다:
  - 군집별 슛 0회 / 패스 0회 경기 비율 — 그 군집이 "스타일"이 아니라 "게임을
    제대로 안 한 사람들"이 우연히 뭉친 것일 가능성 체크
  - 군집별 평균 division(티어) — 군집화 feature에는 안 쓰지만, 특정 군집이 특정
    티어에 쏠려 있으면 "스타일"이 아니라 "실력" 차이를 잡아낸 것일 수 있음 (참고용)
  - 군집별 승/무/패 비율 — 스타일과 승률이 실제로 분리된 개념인지 참고로 같이 확인

train_style.py의 MODEL_SPECS(패스/헤딩/슛위치 3개 모델) 정의를 그대로 재사용해 모델이
늘거나 줄어도 이 스크립트를 따로 고칠 필요가 없게 했다(2026-09-08, 슛 모델이 헤딩축/
슛위치축으로 분리되면서 갱신).

train_style.py를 먼저 실행해 user_style_profile.csv가 생성되어 있어야 한다.

실행: ./venv/Scripts/python.exe style/check_cluster_quality.py
"""

import os

import pandas as pd

import preprocess_style as preprocess
import train_style as train
import utils

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_FILE = os.path.join(REPO_ROOT, "data", "style", "user_style_profile.csv")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "style")
LOW_ACTIVITY_SHOOT_THRESHOLD_PCT = 5.0  # 슛 0회 경기 비율이 이 값(%)을 넘으면 요주의 표시
# ====================================================


def build_cluster_quality_report(matches, profile_df, label_column):
    """label_column(pass_style_type/heading_style_type/shot_location_style_type)
    기준으로 원본 경기 데이터를 다시 이어붙여 군집별 데이터 품질 지표를 낸다.

    몰수/오류 경기는 preprocess_style.py의 군집화 대상 집계와 동일하게 제외한다.
    """
    normal = matches[matches[preprocess.MATCH_END_TYPE_COL] == preprocess.NORMAL_MATCH_END_TYPE]
    merged = normal.merge(
        profile_df[["ouid", label_column]], on=preprocess.OUID_COL, how="inner"
    )

    rows = []
    for cluster_id, group in merged.groupby(label_column):
        rows.append({
            "cluster": cluster_id,
            "n_matches": len(group),
            "n_users": group[preprocess.OUID_COL].nunique(),
            "avg_division": group["division"].mean(),
            "no_shoot_pct": (group["shootTotal"] == 0).mean() * 100,
            "no_pass_pct": (group["passTry"] == 0).mean() * 100,
            "win_pct": (group["matchResult"] == "승").mean() * 100,
            "draw_pct": (group["matchResult"] == "무").mean() * 100,
            "loss_pct": (group["matchResult"] == "패").mean() * 100,
        })
    return pd.DataFrame(rows)


def build_summary_text(model_reports):
    """model_reports: [(key, report_df), ...] (train.MODEL_SPECS 순서)."""
    lines = [
        "=== 스타일 진단 군집 품질 사후 검증 (원본 데이터 드릴다운) ===",
        "군집 평균 feature 값만으로는 '스타일 차이'와 '저활동/이상치'를 구분할 수 없어,",
        "군집 라벨을 원본 경기 데이터(match_team_data.csv)에 다시 연결해 확인한다.",
    ]

    for key, report in model_reports:
        lines.append("")
        lines.append(f"--- {key} 모델 ---")
        lines.append(report.round(2).to_string(index=False))

    lines.append("")
    lines.append(
        f"[요주의 기준] 슛 0회 경기 비율이 {LOW_ACTIVITY_SHOOT_THRESHOLD_PCT}%를 넘는 군집은 "
        "'슛 스타일'이 아니라 '저활동/이상치'로 해석될 여지가 있어 별도 확인이 필요하다."
    )

    for key, report in model_reports:
        flagged = report[report["no_shoot_pct"] > LOW_ACTIVITY_SHOOT_THRESHOLD_PCT]
        if not flagged.empty:
            lines.append(
                f"[요주의] {key} 모델 군집 {list(flagged['cluster'])}: "
                "슛 0회 경기 비율이 기준을 초과함"
            )
        else:
            lines.append(f"[정상] {key} 모델은 모든 군집에서 슛 0회 경기 비율이 기준 이내")

    lines.append(
        "[참고] avg_division(티어)이 군집 간 크게 벌어져 있으면, 이 군집 차이가 "
        "'스타일'이 아니라 '실력'을 반영했을 가능성도 함께 고려해야 한다 "
        "(data-schema.md: long/through_pass_ratio가 division과 상관관계가 있음이 이미 확인됨)."
    )
    return "\n".join(lines)


def main():
    if not os.path.exists(PROFILE_FILE):
        raise FileNotFoundError(
            f"{PROFILE_FILE} 이 없다 - 먼저 style/train_style.py를 실행해 "
            "user_style_profile.csv를 생성해야 한다."
        )

    print("1) 원본 경기 데이터 + 유저별 군집 라벨 로드")
    matches = preprocess.load_matches(preprocess.MATCHES_FILE)
    profile_df = pd.read_csv(PROFILE_FILE)

    model_reports = []
    for i, (key, _, label_column) in enumerate(train.MODEL_SPECS, start=2):
        print(f"\n{i}) {key} 모델 군집 품질 검증")
        report = build_cluster_quality_report(matches, profile_df, label_column)
        print(report.round(2).to_string(index=False))
        model_reports.append((key, report))

    summary_text = build_summary_text(model_reports)
    utils.ensure_dir(OUTPUT_DIR)
    utils.save_text(summary_text, os.path.join(OUTPUT_DIR, "cluster_quality_check.txt"))
    print(f"\n[저장] {OUTPUT_DIR}/cluster_quality_check.txt")


if __name__ == "__main__":
    main()
