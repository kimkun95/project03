"""
matches.jsonl / player_stats_final.csv 파싱 및 feature 조립.

matches.jsonl 필드 매핑은 2026-09-04에 실제 API 응답(match-detail)을 1건 조회해 확인한
스키마를 기준으로 한다:
    matchInfo[i]["division"]                       -> tier (정수 코드, 예: 2300, 2400)
    matchInfo[i]["matchDetail"]["matchResult"]      -> "승" / "패" / "무"
    matchInfo[i]["matchDetail"]["matchEndType"]     -> 0이 정상 종료
    matchInfo[i]["player"][j]["spPosition"] == 28   -> 교체선수(SUB), 스쿼드 계산에서 제외

player_stats_final.csv는 2026-09-04에 팀원이 extract_needed_spids.py로 뽑은 spId
2,916개를 API로 조회해 받아온 실물 파일(UTF-8, 2916행, 커버리지 100%)에
2026-09-07 player_stats_final_v2.csv(1,271개, spid 중복 없음)를 병합해 총 4,187행이 된 파일로
CSV_SPID_COLUMN / CSV_STAT_COLUMNS를 확정했다. 컬럼명이 바뀌면 이 파일 상단 CONFIG만
고쳐서 맞추면 되도록 분리해뒀다. 필요한 컬럼이 없으면 조용히 넘어가지 않고 실제 컬럼
목록을 보여주며 에러를 낸다.

이 파일에는 강화단계(spGrade)를 나타내는 컬럼이 없어, 스탯 값이 몇강 기준인지 확인할
방법이 없다. 확인 전까지는 강화단계를 무시하고 spId만으로 매칭하는 근사치로 취급한다
(한계로 명시, train_winrate.build_summary_text()가 쓰는 summary 텍스트에도 남긴다).

포지션 그룹별 능력치(공격/미드필더/수비/GK, Notion ERD 기준)
--------------------------------------------------------
avg_stat_score 하나로 뭉뚱그리던 것을 attack_avg_score / mid_avg_score /
defense_avg_score / gk_avg_score 4개로 나눈다. spPosition 코드(api-constraints.md에서
확인된 28개 코드)로 선수를 4개 그룹으로 나누고, 그룹마다 다른 스탯 subset을 평균한다.

"어떤 스탯을 특화 스탯으로 볼지"는 원래 팀 내 미확정 상태였다(ERD 액션아이템 참고).
임의로 정하면 CLAUDE.md의 "가중치 근거 약한 방식 채택 안 함" 원칙에 어긋나므로, 우리가
직접 고르는 대신 EA/FC 온라인이 실제로 쓰는 공식 6분류(스피드/슈팅/패스/드리블/수비/
피지컬, 실제 게임 UI의 PAC/SHO/PAS/DRI/DEF/PHY)를 그대로 가져와 근거로 삼는다:
    공격 = 스피드 + 드리블 + 피지컬 + 슈팅(특화)
    수비 = 스피드 + 드리블 + 피지컬 + 수비(특화)
    미드필더 = 6개 카테고리 전체(29개 스탯 그대로) — 확정된 것 유지
    GK = GK 전용 5개 — 확정된 것 유지
헤더는 공식 분류상 수비(DEF) 카테고리에만 속하지만, 크로스를 받아 헤딩골을 넣는 것도
실제 축구에서 명백한 득점 루트라 팀 논의로 슈팅(SHO) 카테고리에도 중복 포함시키기로
했다 — 공식 분류에서 벗어나는 유일한 예외이며, 그 이유가 명확해 임의 가중치 문제로
보지 않는다. 그룹 내부에서는 여전히 전부 동일 가중치로 평균한다(가중치를 준 게 아니라
"어떤 스탯이 그 포지션과 관련 있는가"만 골랐을 뿐이다).
"""

import json
import os

import numpy as np
import pandas as pd

# ============ CONFIG ============
# 실행 위치(cwd)에 관계없이 항상 저장소 루트 기준 data/winrate를 가리키도록
# 스크립트 파일 위치에서 경로를 계산한다 (winrate/ 밑에서 실행해도 안전).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_FILE = os.path.join(REPO_ROOT, "data", "winrate", "matches.jsonl")
PLAYER_CARD_FILE = os.path.join(REPO_ROOT, "data", "player_stats_final.csv")

# 승패와 사실상 동일한 정보(결과)가 feature에 섞여 들어가는 것을 막기 위한 금지 키워드.
# CLAUDE.md의 leakage 원칙(슛수/골수/태클 성공률/경기평점 등 결과 관련 필드 금지)을 그대로 반영.
LEAKAGE_FORBIDDEN_KEYWORDS = [
    "goal", "shoot", "슛", "골", "rating", "tackle", "foul", "card",
    "offside", "controller", "penalty", "freekick",
]

# --- matches.jsonl 필드 매핑 (2026-09-04 실제 API 응답으로 확인됨) ---
SUB_POSITION_CODE = 28
WIN_LABEL = "승"
LOSE_LABEL = "패"
DRAW_LABEL = "무"
NORMAL_MATCH_END_TYPE = 0

# --- 포지션 그룹 분류 (spPosition 코드, .claude/rules/api-constraints.md에서 확인됨) ---
# GK=0. SUB(28)은 extract_rows 단계에서 이미 제외되므로 여기 포함하지 않는다.
GK_POSITION_CODE = 0
POSITION_GROUP_DEFENSE = {1, 2, 3, 4, 5, 6, 7, 8}       # SW,RWB,RB,RCB,CB,LCB,LB,LWB
POSITION_GROUP_MIDFIELD = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}  # RDM~LAM
POSITION_GROUP_ATTACK = {20, 21, 22, 23, 24, 25, 26, 27}  # RF~LW

# --- player_stats_final.csv 컬럼 매핑 (2026-09-04 팀원이 우리 needed_spids.json 2,916개로
# API 조회해 받아온 실물 파일로 확인됨, UTF-8, spid 커버리지 100%; 2026-09-07 v2 1,271개
# 병합으로 현재 4,187행, spid 중복 없음 확인됨) ---
# 강화단계(spGrade)를 나타내는 컬럼이 파일에 없어, 이 스탯 값이 몇강 기준인지 확인 불가.
# 확인 전까지는 강화단계를 무시하고 spId만으로 매칭하는 근사치로 취급한다 (한계로 명시).
CSV_SPID_COLUMN = "spid"
CSV_STAT_COLUMNS = [
    "속력", "가속력", "골 결정력", "슛 파워", "중거리 슛", "위치 선정", "발리슛", "페널티 킥",
    "짧은 패스", "시야", "크로스", "긴 패스", "프리킥", "커브", "드리블", "볼 컨트롤",
    "민첩성", "밸런스", "반응 속도", "대인 수비", "태클", "가로채기", "헤더", "슬라이딩 태클",
    "몸싸움", "스태미너", "적극성", "점프", "침착성",
]
# GK 전용 5개 스탯 (공격/미드필더/수비 그룹에는 쓰지 않고, GK 그룹 평균에만 사용).
CSV_GK_STAT_COLUMNS = ["GK 다이빙", "GK 핸들링", "GK 킥", "GK 반응속도", "GK 위치 선정"]

# --- EA/FC 온라인 공식 6분류 (게임 UI의 PAC/SHO/PAS/DRI/DEF/PHY) ---
# CSV_STAT_COLUMNS 29개를 이 6개 카테고리로 전부 나눈다 (2+7+6+6+5+4=30 — 헤더가
# SHO/DEF 양쪽에 중복 포함되는 예외 하나 때문에 29가 아니라 30으로 합이 늘어난다).
CAT_PAC = ["속력", "가속력"]
CAT_SHO = ["골 결정력", "슛 파워", "중거리 슛", "위치 선정", "발리슛", "페널티 킥", "헤더"]
CAT_PAS = ["짧은 패스", "시야", "크로스", "긴 패스", "프리킥", "커브"]
CAT_DRI = ["드리블", "볼 컨트롤", "민첩성", "밸런스", "반응 속도", "침착성"]
CAT_DEF = ["대인 수비", "태클", "가로채기", "헤더", "슬라이딩 태클"]
CAT_PHY = ["몸싸움", "스태미너", "적극성", "점프"]

# 그룹별 사용 스탯 (모듈 docstring의 "포지션 그룹별 능력치" 설명 참고).
ATTACK_STAT_COLUMNS = CAT_PAC + CAT_DRI + CAT_PHY + CAT_SHO
DEFENSE_STAT_COLUMNS = CAT_PAC + CAT_DRI + CAT_PHY + CAT_DEF
MID_STAT_COLUMNS = CSV_STAT_COLUMNS  # 확정된 것 유지: 6개 카테고리 전체(29개)
# ====================================================


def load_matches(path):
    """matches.jsonl을 한 줄씩 읽어 dict 리스트로 반환한다."""
    matches = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                matches.append(json.loads(line))
    return matches


def extract_rows(matches):
    """matches.jsonl의 raw match-detail 리스트에서 (match_id, ouid) 단위 학습용 행을 뽑는다.

    매치 하나(matchInfo 2건, 양 팀)에서 최대 2행을 추출한다. matchEndType != 0(몰수경기),
    matchResult == "무"(무승부)는 제외하고, 같은 ouid가 여러 match_id에 걸쳐 중복 등장하면
    첫 번째만 남긴다. 각 필터링 사유별 제외 건수를 로그로 남긴다.
    """
    rows = []
    excluded_forfeit = 0
    excluded_draw = 0
    seen_ouid = set()
    excluded_dup_ouid = 0

    for match in matches:
        match_id = match.get("matchId")
        for info in match.get("matchInfo", []):
            match_detail = info.get("matchDetail", {})

            if match_detail.get("matchEndType") != NORMAL_MATCH_END_TYPE:
                excluded_forfeit += 1
                continue

            result_label = match_detail.get("matchResult")
            if result_label == DRAW_LABEL:
                excluded_draw += 1
                continue
            if result_label not in (WIN_LABEL, LOSE_LABEL):
                continue

            ouid = info.get("ouid")
            if ouid in seen_ouid:
                excluded_dup_ouid += 1
                continue
            seen_ouid.add(ouid)

            sp_players = [
                {"sp_id": p.get("spId"), "sp_position": p.get("spPosition")}
                for p in info.get("player", [])
                if p.get("spPosition") != SUB_POSITION_CODE
            ]

            rows.append({
                "match_id": match_id,
                "ouid": ouid,
                "tier": info.get("division"),
                "sp_players": sp_players,
                "result": 1 if result_label == WIN_LABEL else 0,
            })

    print(f"  [필터링] 몰수경기 제외: {excluded_forfeit}건")
    print(f"  [필터링] 무승부 제외: {excluded_draw}건")
    print(f"  [필터링] 중복 ouid 제외: {excluded_dup_ouid}건")
    print(f"  [추출] 최종 행 수: {len(rows)}")
    return pd.DataFrame(rows)


def load_player_cards(csv_path):
    """PLAYER_CARD csv를 로드하고, 필요한 컬럼이 실제로 있는지 검증한다."""
    df = pd.read_csv(csv_path)
    required = [CSV_SPID_COLUMN] + CSV_STAT_COLUMNS + CSV_GK_STAT_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path}에 필요한 컬럼이 없습니다: {missing}\n"
            f"실제 컬럼 목록: {list(df.columns)}\n"
            "preprocess_winrate.py 상단 CONFIG의 CSV_SPID_COLUMN/CSV_STAT_COLUMNS/"
            "CSV_GK_STAT_COLUMNS를 실제 파일에 맞게 수정하세요."
        )
    return df.set_index(CSV_SPID_COLUMN)


def classify_position_group(sp_position):
    """spPosition 코드를 attack/midfield/defense/gk 중 하나로 분류한다. 미상이면 None."""
    if sp_position == GK_POSITION_CODE:
        return "gk"
    if sp_position in POSITION_GROUP_DEFENSE:
        return "defense"
    if sp_position in POSITION_GROUP_MIDFIELD:
        return "midfield"
    if sp_position in POSITION_GROUP_ATTACK:
        return "attack"
    return None


_GROUP_TO_FEATURE_COLUMN = {
    "attack": "attack_avg_score",
    "midfield": "mid_avg_score",
    "defense": "defense_avg_score",
    "gk": "gk_avg_score",
}
_GROUP_STAT_COLUMNS = {
    "attack": ATTACK_STAT_COLUMNS,
    "midfield": MID_STAT_COLUMNS,
    "defense": DEFENSE_STAT_COLUMNS,
    "gk": CSV_GK_STAT_COLUMNS,
}


def compute_group_avg_scores(sp_players, player_card_df):
    """spId+spPosition 목록을 포지션 그룹별로 나눠 그룹별 평균 스탯을 낸다.

    그룹마다 다른 스탯 subset을 동일 가중치로 평균한다(모듈 docstring 참고): 공격/수비는
    EA FC 공식 카테고리(스피드+드리블+피지컬+특화 카테고리)를, 미드필더는 29개 전체를,
    GK는 GK 전용 5개를 쓴다. 매칭 안 되는 spId는 평균에서 제외하고, 그룹에 매칭된 선수가
    하나도 없으면 그 그룹 점수는 NaN.
    반환값: (그룹별 점수 dict, 조회 시도한 spId 수, 매칭된 spId 수) — 시도/매칭 수는 4개
    그룹 합산.
    """
    grouped_sp_ids = {"attack": [], "midfield": [], "defense": [], "gk": []}
    for p in sp_players:
        group = classify_position_group(p["sp_position"])
        if group:
            grouped_sp_ids[group].append(p["sp_id"])

    scores = {}
    total_attempted = 0
    total_matched = 0
    for group, sp_ids in grouped_sp_ids.items():
        stat_columns = _GROUP_STAT_COLUMNS[group]
        matched_scores = []
        for sp_id in sp_ids:
            if sp_id in player_card_df.index:
                row = player_card_df.loc[sp_id, stat_columns]
                matched_scores.append(float(np.mean(row.values.astype(float))))

        total_attempted += len(sp_ids)
        total_matched += len(matched_scores)
        scores[_GROUP_TO_FEATURE_COLUMN[group]] = (
            float(np.mean(matched_scores)) if matched_scores else np.nan
        )

    return scores, total_attempted, total_matched


GROUP_SCORE_COLUMNS = list(_GROUP_TO_FEATURE_COLUMN.values())


def assemble_feature_table(rows_df, player_card_df):
    """extract_rows 결과에 포지션 그룹별 avg_score 4개를 붙여 최종 feature 테이블을 만든다."""
    score_rows = []
    total_attempted = 0
    total_matched = 0

    for sp_players in rows_df["sp_players"]:
        scores, attempted, matched = compute_group_avg_scores(sp_players, player_card_df)
        score_rows.append(scores)
        total_attempted += attempted
        total_matched += matched

    rows_df = rows_df.copy()
    scores_df = pd.DataFrame(score_rows, index=rows_df.index)
    rows_df[GROUP_SCORE_COLUMNS] = scores_df[GROUP_SCORE_COLUMNS]

    match_rate = (total_matched / total_attempted * 100) if total_attempted else 0.0
    print(f"  [avg_score] spId 매칭률: {match_rate:.1f}% ({total_matched}/{total_attempted})")

    before = len(rows_df)
    feature_df = rows_df.dropna(subset=GROUP_SCORE_COLUMNS + ["tier"]).copy()
    dropped = before - len(feature_df)
    print(f"  [avg_score] 그룹 중 하나라도 매칭된 선수가 없거나 tier 결측인 행 제외: {dropped}건")

    return (
        feature_df[["match_id", "ouid"] + GROUP_SCORE_COLUMNS + ["tier", "result"]],
        match_rate,
    )


def assert_no_leakage(feature_columns):
    """feature 목록에 결과와 사실상 동일한 정보(골/슛/평점 등)가 없는지 확인한다."""
    for col in feature_columns:
        col_lower = col.lower()
        for keyword in LEAKAGE_FORBIDDEN_KEYWORDS:
            assert keyword.lower() not in col_lower, (
                f"leakage 의심: feature '{col}'에 금지 키워드 '{keyword}'가 포함되어 있습니다."
            )
