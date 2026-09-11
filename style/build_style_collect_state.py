"""
matches_style.jsonl에 이미 들어있는 경기/유저 정보로 collect_state_style.json을 재구성한다.

collect_style_matches.py는 STATE_FILE(collect_state_style.json)이 없으면 seen_match/
seen_ouid를 빈 상태로 시작한다. 그런데 matches_style.jsonl에는 이미 2026-09-07에
수집해둔 4,080경기가 들어있어서(팀원 데이터를 받기 전에 직접 모은 것), 이 상태로 그냥
다시 실행하면:
  - 이미 가진 유저를 다시 API로 조회해 호출을 낭비하고
  - 같은 매치를 matches_style.jsonl에 중복으로 append하게 된다.

이 스크립트는 matches_style.jsonl을 한 번 읽어서 이미 등장한 matchId 전부를
seen_match로, matchInfo에 등장한 ouid/nickname 전부를 seen_ouid/queued_nicknames로
채운 state 파일을 만든다. queue는 SEED_NICKNAMES 그대로 둔다 — 이미 처리된 유저라도
get_ouid() 호출 1번(닉네임 -> ouid)만 하면 seen_ouid에 걸려 곧바로 건너뛰므로, 호출
낭비 없이 안전하게 다시 시드부터 시작할 수 있다.

이미 seen_ouid로 표시된 유저는(원래 primary로 처리됐든, 상대방으로 스쳐 지나갔든 구분
없이) 이번 재구성에서는 전부 "이미 다뤘음"으로 취급한다 — 지금 목표가 기존 유저를 더
깊게 파는 게 아니라 표본(유저 수) 자체를 넓히는 것이므로, 스노볼이 같은 유저를 다시
큐에 넣지 않게 하는 쪽이 API 예산을 새 유저 확보에 더 많이 쓸 수 있어 유리하다.

실행 전 조건: matches_style.jsonl은 있어야 하고, collect_state_style.json은 없어야
한다(이미 있으면 기존 진행 상태를 실수로 덮어쓰지 않도록 중단한다).

실행: ./venv/Scripts/python.exe style/build_style_collect_state.py
"""

import json
import os

from collect_style_matches import OUT_FILE, SEED_NICKNAMES, STATE_FILE


def build_state_from_jsonl(path):
    """matches_style.jsonl을 순회하며 seen_match/seen_ouid/queued_nicknames 후보를 뽑는다."""
    seen_match = set()
    seen_ouid = set()
    queued_nicknames = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = json.loads(line)

            match_id = match.get("matchId")
            if match_id:
                seen_match.add(match_id)

            for info in match.get("matchInfo", []):
                ouid = info.get("ouid")
                nickname = info.get("nickname")
                if ouid:
                    seen_ouid.add(ouid)
                if nickname:
                    queued_nicknames.add(nickname)

    return seen_match, seen_ouid, queued_nicknames


def main():
    if not os.path.exists(OUT_FILE):
        raise FileNotFoundError(f"{OUT_FILE} 이 없다 - 재구성할 기존 데이터가 없다.")
    if os.path.exists(STATE_FILE):
        raise FileExistsError(
            f"{STATE_FILE} 이 이미 있다 - 기존 진행 상태를 실수로 덮어쓰지 않기 위해 "
            "중단한다. 새로 재구성하려면 기존 파일을 먼저 지우거나 옮겨야 한다."
        )

    print(f"1) {OUT_FILE} 로드 및 기존 matchId/ouid/nickname 추출")
    seen_match, seen_ouid, queued_nicknames = build_state_from_jsonl(OUT_FILE)
    print(f"  기존 경기 수(seen_match): {len(seen_match)}")
    print(f"  기존 고유 유저 수(seen_ouid): {len(seen_ouid)}")

    queued_nicknames |= set(SEED_NICKNAMES)

    state = {
        "queue": list(SEED_NICKNAMES),
        "seen_ouid": sorted(seen_ouid),
        "seen_match": sorted(seen_match),
        "queued_nicknames": sorted(queued_nicknames),
    }

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print(f"2) 저장 완료: {STATE_FILE}")
    print(f"  다음 실행 시 큐(시드부터 재시작): {state['queue']}")


if __name__ == "__main__":
    main()
