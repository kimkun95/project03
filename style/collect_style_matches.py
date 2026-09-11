"""
스타일진단용 매치 데이터 수집 스크립트 (유저당 최근 최대 100경기, 깊게 수집)

승률 트랙(winrate/snowball_collect.py)과 API/저장 스키마는 동일하지만 수집 "로직"은
다르다 (CLAUDE.md 2트랙 전략 참고):
    - 승률 트랙: 유저당 1경기만, 최대한 많은 유저로 넓게
    - 스타일 트랙: 유저당 최근 최대 100경기까지 깊게 (한 유저의 플레이 성향을
      안정적으로 추정하려면 표본이 여러 건 필요하기 때문)

한 유저를 완전히 수집하려면 매치 목록 조회 1회 + 매치 상세 조회 최대 100회, 총 최대
101회가 든다. 일일 호출 한도(개발 단계 1,000건)를 감안하면 하루에 완전히 수집 가능한
유저는 최대 9~10명 수준이다 — 이 스크립트는 하루 예산 안에서 유저를 하나씩 "완결"시켜
가며(중간에 끊기지 않게) 진행하고, 남은 예산이 다음 유저 하나를 다 못 채울 것 같으면
그 유저는 아예 시작하지 않고 다음 실행으로 미룬다.

저장은 승률 트랙과 동일하게 API 응답을 가공 없이 통째로 matches_style.jsonl에 남긴다
(원칙은 CLAUDE.md "저장 방침" 참고). 시드 유저만으로는 스타일 다양성이 부족할 수 있어,
승률 트랙처럼 매치 상세 응답 속 상대방 닉네임으로 큐를 넓혀가는 스노볼 방식도 함께 쓴다.

실행 전에 아래 CONFIG 섹션만 채우면 된다.
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

# ============ CONFIG ============
# API_KEY는 .env 파일의 NEXON_API_KEY에서 자동으로 채워짐 (여기서 직접 수정하지 말 것)
API_KEY = os.environ.get("NEXON_API_KEY", "여기에_발급받은_API_키")
# 아래부터 실행 전에 직접 채워야 하는 값. winrate 트랙과 같은 시드를 재사용해도 되고
# (CLAUDE.md: 두 트랙에 유저가 겹쳐도 문제없음), 스타일 진단은 표본 다양성이 더 중요하니
# 팀원 닉네임 등으로 자유롭게 바꿔도 된다.
SEED_NICKNAMES = ["피파재미없음", "경품싹쓸이", "감형", "준서누뗄라이련아", "당신은승리자"]
MATCHTYPE = 50  # 공식경기 (winrate/snowball_collect.py에서 확정된 값과 동일)
MAX_MATCHES_PER_USER = 100  # api-constraints.md: /user/match limit 최대값
# match_team_data.csv 실측 기준 몰수/오류 경기 비율이 22.5%였다(2026-09-08 확인).
# 학습 하한선을 아직 50/60/70 중 뭘로 정할지 팀 결정 전이라(2026-09-08 기준 열린 논의),
# 어느 쪽으로 정해지든 다시 수집할 필요 없게 가장 보수적인 쪽으로 넉넉히 모아둔다.
# "정상종료 70경기"를 기대값 수준에서 넘기려면 원본 매치가 최소 70/(1-0.225)=90.3개는
# 있어야 한다(정상종료 기대값 = 원본 개수 x (1 - 0.225)). MAX_MATCHES_PER_USER(=100)에
# 근접한 유저만 사실상 통과하는 셈이라 자격 유저 풀은 확 줄어들지만, 그만큼 수집된
# 유저는 학습 하한선이 나중에 70으로 정해져도 안전하다.
MIN_SAMPLE_MATCHES = 90
DAILY_CALL_BUDGET = 950  # 일일 한도 1,000 중 여유 100 남김
REQUEST_INTERVAL = 0.35  # 초당 5건 제한 대응

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(REPO_ROOT, "data", "style", "collect_state_style.json")
OUT_FILE = os.path.join(REPO_ROOT, "data", "style", "matches_style.jsonl")
# ====================================================

BASE_URL = "https://open.api.nexon.com/fconline/v1"
HEADERS = {"x-nxopen-api-key": API_KEY}

call_count = 0


class RateLimitedError(Exception):
    """429 등 레이트리밋/일일 한도 초과. 큐에서 버리지 않고 되돌린 뒤 즉시 멈춘다."""


def _get(path, params):
    global call_count
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
    call_count += 1
    time.sleep(REQUEST_INTERVAL)
    if resp.status_code == 429:
        raise RateLimitedError(f"{path} 429: {resp.text[:200]}")
    if resp.status_code != 200:
        print(f"  [경고] {path} 호출 실패 ({resp.status_code}): {resp.text[:200]}")
        return None
    return resp.json()


def get_ouid(nickname):
    data = _get("/id", {"nickname": nickname})
    if data and "ouid" in data:
        return data["ouid"]
    return None


def get_match_list(ouid, matchtype, limit=MAX_MATCHES_PER_USER):
    data = _get("/user/match", {"ouid": ouid, "matchtype": matchtype, "offset": 0, "limit": limit})
    return data or []


def get_match_detail(match_id):
    return _get("/match-detail", {"matchid": match_id})


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        state["queue"] = list(dict.fromkeys(state.get("queue", [])))
        state.setdefault("queued_nicknames", list(state["queue"]))
        return state
    return {
        "queue": list(SEED_NICKNAMES),
        "seen_ouid": [],
        "seen_match": [],
        "queued_nicknames": list(SEED_NICKNAMES),
    }


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def main():
    state = load_state()
    seen_ouid = set(state["seen_ouid"])
    seen_match = set(state["seen_match"])
    queued_nicknames = set(state["queued_nicknames"])
    queue = state["queue"]

    users_collected_this_run = 0
    matches_collected_this_run = 0
    rate_limited = False

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "a", encoding="utf-8") as out:
        while queue:
            # 유저 한 명을 끝까지(최대 101회 호출) 못 채울 것 같으면 예산을 아껴 다음
            # 실행으로 미룬다 — 중간에 끊겨 한 유저의 표본이 어중간하게 남는 것을 피한다.
            if call_count + MAX_MATCHES_PER_USER + 1 > DAILY_CALL_BUDGET:
                print("  [예산 부족] 유저 한 명을 다 채울 예산이 안 남아 다음 실행으로 미룬다.")
                break

            nickname = queue.pop(0)

            try:
                ouid = get_ouid(nickname)
                if not ouid or ouid in seen_ouid:
                    continue
                seen_ouid.add(ouid)

                match_ids = get_match_list(ouid, MATCHTYPE, limit=MAX_MATCHES_PER_USER)
                if not match_ids:
                    continue
                if len(match_ids) < MIN_SAMPLE_MATCHES:
                    # 공식경기 자체가 학습 하한선(MIN_SAMPLE_MATCHES) 미만이면 나중에
                    # preprocess_style.py에서 어차피 전량 제외되므로, match-detail을
                    # 하나도 호출하지 않고 바로 넘어가 API 호출을 아낀다.
                    print(f"  [건너뜀] {nickname}: 공식경기 {len(match_ids)}건 "
                          f"(최소 {MIN_SAMPLE_MATCHES}건 미만이라 학습에 못 씀)")
                    continue

                saved_for_user = 0
                for match_id in match_ids:
                    if call_count >= DAILY_CALL_BUDGET - 3:
                        break
                    if not match_id or match_id in seen_match:
                        continue

                    detail = get_match_detail(match_id)
                    if not detail:
                        continue

                    # 스노볼 확장: 상대방 닉네임을 큐에 추가 (표본 다양성 확보용)
                    for info in detail.get("matchInfo", []):
                        opp_ouid = info.get("ouid")
                        opp_nick = info.get("nickname")
                        if opp_ouid and opp_ouid not in seen_ouid and opp_nick and opp_nick not in queued_nicknames:
                            queue.append(opp_nick)
                            queued_nicknames.add(opp_nick)

                    seen_match.add(match_id)
                    out.write(json.dumps(detail, ensure_ascii=False) + "\n")
                    saved_for_user += 1
                    matches_collected_this_run += 1

                if saved_for_user > 0:
                    users_collected_this_run += 1
                    print(f"  [완료] {nickname}: {saved_for_user}경기 저장, 누적 API 호출 {call_count}회")
            except RateLimitedError as e:
                queue.insert(0, nickname)
                print(f"\n  [중단] 레이트리밋 감지, 실행을 멈춘다: {e}")
                rate_limited = True
                break

    state["queue"] = queue
    state["seen_ouid"] = list(seen_ouid)
    state["seen_match"] = list(seen_match)
    state["queued_nicknames"] = list(queued_nicknames)
    save_state(state)

    print("\n=== 이번 실행 요약 ===")
    print(f"수집 완료한 유저 수: {users_collected_this_run}")
    print(f"수집한 경기 수: {matches_collected_this_run}")
    print(f"API 호출 횟수: {call_count}")
    print(f"누적 고유 유저 수: {len(seen_ouid)}")
    print(f"누적 고유 경기 수: {len(seen_match)}")
    print(f"다음 실행 대기 큐: {len(queue)}명")
    print(f"저장 파일: {OUT_FILE} (누적 append 방식)")
    if rate_limited:
        print("레이트리밋(429)으로 중단됨 — 일일 한도가 다 찼을 가능성이 높다. 시간을 두고 다시 실행할 것.")


if __name__ == "__main__":
    main()
