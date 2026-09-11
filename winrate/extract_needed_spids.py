"""
matches.jsonl에 실제로 등장한 고유 spId를 뽑아 저장한다.

player_1000_final.csv(팀원이 API로 받아온 고정 1,000개 카드 목록)의 매칭률이 낮아서
(data-schema.md 미해결 항목 참고), 우리가 실제로 쓴 카드 spId를 직접 확보해 팀원의
API로 개별 조회하는 수집 스크립트를 만들 예정이다. 이 스크립트는 그 조회 대상 목록을
만드는 용도다.

spPosition == 28(SUB, 교체선수)은 스쿼드 계산에서 제외하는 기존 원칙(data-schema.md)을
그대로 따른다.
"""

import json
import os

# ============ CONFIG ============
# 실행 위치(cwd)에 관계없이 항상 저장소 루트 기준 경로를 쓰도록 스크립트 파일 위치에서 계산한다.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_FILE = os.path.join(REPO_ROOT, "data", "winrate", "matches.jsonl")
OUTPUT_FILE = os.path.join(REPO_ROOT, "data", "winrate", "needed_spids.json")
SUB_POSITION_CODE = 28
# ====================================================


def extract_needed_spids(matches_file):
    """matches.jsonl 전체를 훑어 SUB 제외 고유 spId 목록을 오름차순으로 반환한다."""
    sp_ids = set()
    with open(matches_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = json.loads(line)
            for info in match.get("matchInfo", []):
                for p in info.get("player", []):
                    if p.get("spPosition") != SUB_POSITION_CODE:
                        sp_ids.add(p.get("spId"))
    return sorted(sp_ids)


def main():
    sp_ids = extract_needed_spids(MATCHES_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sp_ids, f)
    print(f"고유 spId {len(sp_ids)}개 -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
