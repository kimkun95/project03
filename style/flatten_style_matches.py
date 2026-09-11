"""
matches_style.jsonl(raw match-detail API 응답, 매치당 1줄) -> match_team_data.csv와
동일한 스키마의 팀 단위 평탄화 CSV로 변환한다.

data/style/raw_data/(팀원이 아닌 우리가 직접 수집한 매치 상세 응답 3,993건)를
matches_style.jsonl로 합친 뒤(2026-09-07), 이 raw 데이터도 preprocess_style.py
파이프라인에서 쓸 수 있게 하려고 만들었다. preprocess_style.py는 원래
"팀원이 이미 팀 단위로 평탄화해 수집한 match_team_data.csv"를 원본으로 쓰기로
결정된 상태라(2026-09-07), 그 팀원 파일과 같은 컬럼/순서를 그대로 재현해 호환되게
만든다 — 그래야 두 파일을 이어붙이거나(pd.concat), preprocess_style.py의
MATCHES_FILE 경로만 바꿔서 그대로 재사용할 수 있다.

⚠️ match_team_data.csv(팀원 소유 파일)는 건드리지 않는다. 이 스크립트는 별도
파일(OUT_FILE)에만 쓴다 — 두 파일을 합칠지 여부는 팀 논의 후 결정할 일이다.

매치 하나(matchInfo 2건, 양 팀 시점)에서 최대 2행을 뽑는다. match_team_data.csv와
동일하게 이 단계에서는 matchEndType으로 거르지 않는다 — 필터링은 다운스트림인
preprocess_style.extract_match_style_rows()의 책임이다(CLAUDE.md "저장 방침":
가공 없이 통째로 저장, 필터링은 가공 단계에서).
"""

import json
import os

import pandas as pd

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_FILE = os.path.join(REPO_ROOT, "data", "style", "matches_style.jsonl")
OUT_FILE = os.path.join(REPO_ROOT, "data", "style", "match_team_data_raw.csv")

# match_team_data.csv와 완전히 동일한 컬럼 이름/순서 (2026-09-07 실제 파일 헤더로 확인).
MATCH_DETAIL_COLUMNS = [
    "seasonId", "matchResult", "matchEndType", "systemPause", "foul", "injury",
    "redCards", "yellowCards", "dribble", "cornerKick", "possession",
    "offsideCount", "averageRating", "controller",
]
SHOOT_COLUMNS = [
    "shootTotal", "effectiveShootTotal", "shootOutScore", "goalTotal",
    "goalTotalDisplay", "ownGoal", "shootHeading", "goalHeading",
    "shootFreekick", "goalFreekick", "shootInPenalty", "goalInPenalty",
    "shootOutPenalty", "goalOutPenalty", "shootPenaltyKick", "goalPenaltyKick",
]
PASS_COLUMNS = [
    "passTry", "passSuccess", "shortPassTry", "shortPassSuccess",
    "longPassTry", "longPassSuccess", "bouncingLobPassTry", "bouncingLobPassSuccess",
    "drivenGroundPassTry", "drivenGroundPassSuccess", "throughPassTry",
    "throughPassSuccess", "lobbedThroughPassTry", "lobbedThroughPassSuccess",
]
DEFENCE_COLUMNS = ["blockTry", "blockSuccess", "tackleTry", "tackleSuccess"]

OUTPUT_COLUMNS = (
    ["matchId", "matchDate", "matchType", "ouid", "nickname", "division"]
    + MATCH_DETAIL_COLUMNS + SHOOT_COLUMNS + PASS_COLUMNS + DEFENCE_COLUMNS
)
# ====================================================


def load_matches(path):
    """matches_style.jsonl을 한 줄씩 읽어 dict 리스트로 반환한다."""
    matches = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                matches.append(json.loads(line))
    return matches


def flatten_matches(matches):
    """raw match-detail 리스트 -> match_team_data.csv와 동일한 스키마의 팀 단위 행 리스트.

    matchInfo가 2건이 아닌(비정상) 매치도 있는 그대로 있는 만큼만 행을 만든다 — 이
    단계에서는 아무것도 거르지 않는다(모듈 docstring 참고).
    """
    rows = []
    for match in matches:
        match_id = match.get("matchId")
        match_date = match.get("matchDate")
        match_type = match.get("matchType")

        for info in match.get("matchInfo", []):
            match_detail = info.get("matchDetail", {}) or {}
            shoot = info.get("shoot", {}) or {}
            pass_ = info.get("pass", {}) or {}
            defence = info.get("defence", {}) or {}

            row = {
                "matchId": match_id,
                "matchDate": match_date,
                "matchType": match_type,
                "ouid": info.get("ouid"),
                "nickname": info.get("nickname"),
                "division": info.get("division"),
            }
            for col in MATCH_DETAIL_COLUMNS:
                row[col] = match_detail.get(col)
            for col in SHOOT_COLUMNS:
                row[col] = shoot.get(col)
            for col in PASS_COLUMNS:
                row[col] = pass_.get(col)
            for col in DEFENCE_COLUMNS:
                row[col] = defence.get(col)

            rows.append(row)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main():
    print(f"1) {MATCHES_FILE} 로드")
    matches = load_matches(MATCHES_FILE)
    print(f"  매치 수: {len(matches)}")

    print("2) 팀 단위 평탄화 (match_team_data.csv와 동일 스키마)")
    df = flatten_matches(matches)
    print(f"  생성된 행 수: {len(df)} (매치당 최대 2행)")
    print("  matchEndType 분포:")
    print(df["matchEndType"].value_counts(dropna=False).to_string())

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"3) 저장 완료: {OUT_FILE}")


if __name__ == "__main__":
    main()
