"""
승률학습용 매치 데이터 수집 스크립트 (스노우볼 방식)

- 시드 닉네임에서 시작해서, 매치 상세 조회 응답에 들어있는 상대방 ouid를
  큐에 계속 추가하며 1유저당 1경기씩 넓게 수집한다.
- 큐 고갈을 완화하기 위해 유저당 최근 QUEUE_EXPAND_LIMIT경기까지 살펴보고 그 안의
  상대방을 전부 큐에 넣는다. 단 matches.jsonl에 저장하는 것은 유저당 1경기(가장 최근 저장 가능한 것)뿐.
- 큐에 넣을 때 이미 처리된 유저(seen_ouid)뿐 아니라 이미 큐에 들어간 적 있는 닉네임(queued_nicknames)도
  걸러서, 같은 사람이 큐에 중복으로 쌓이는 것을 막는다.
- 하루 API 호출 한도(개발단계 1,000건/일) 안에서 자동으로 멈추고,
  진행 상태(queue, seen_ouid, seen_match, seen_detail_match, queued_nicknames)를 파일에 저장해서
  다음 실행 때 이어간다.
- 저장은 API 응답을 통째로(가공/필터링 없이) matches.jsonl에 남긴다.
  jsonl 용량 자체가 무시할 수준이라 저장 시점에 필드를 미리 지울 이유가 없고,
  이미 API 호출 비용을 써서 받은 데이터를 저장 단계에서 버리면 나중에 다른 필드가
  필요해졌을 때 재수집(=API 호출 재낭비)해야 한다. player[].status나 shootDetail처럼
  안 쓰는 필드를 골라내는 작업은 CSV/학습 데이터로 가공하는 전처리 스크립트에서
  필요한 필드만 뽑아 쓰는 방식으로 처리한다 (저장 단계가 아니라 가공 단계의 책임).

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
# 아래부터 실행 전에 직접 채워야 하는 값
SEED_NICKNAMES = ["바람과함께살빼다", "고대넘버원호동생", "내일모레의용재시","박지성과퍼거슨","진짜닌자", "MOSTPOP",
                   "양준근이", "태림이", "톰상이", "경중선", "리오넬보아탱", "brianeelee", "소울닌자", "3사단화지대", "김태호발닦개",
                   "우유맛볻앵", "아침작은새", "쿠마이누", "세브첸코85", "엘에프씨"]
MATCHTYPE = 50  # 확정됨: "공식경기" (1대1 랭크 매치). /metadata/matchtype 전체 목록에서
# 30=리그 친선, 40=클래식 1on1, 50=공식경기, 52=감독모드, 60=공식 친선,
# 204/214/224/234=볼타 계열(3v3) 중 "1대1 공식경기" 요건에 정확히 부합하는 것은 50뿐.
DAILY_CALL_BUDGET = 950  # 일일 한도 1,000 중 여유 100 남김
REQUEST_INTERVAL = 0.5  # 초당 5건 제한 대응 (0.25초는 예산 소진 직전 429 발생해 0.35초로 늘림)
QUEUE_EXPAND_LIMIT = 1  # 큐 확장용으로 살펴볼 유저당 최근 경기 수 (matches.jsonl 저장은 여전히 유저당 1경기)

# 실행 위치(cwd)에 관계없이 항상 저장소 루트 기준 경로를 쓰도록 스크립트 파일 위치에서 계산한다
# (winrate/ 폴더 안에서 실행해도 저장 위치가 갈라지지 않게 하기 위함).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(REPO_ROOT, "data", "winrate", "collect_state.json")
OUT_FILE = os.path.join(REPO_ROOT, "data", "winrate", "matches.jsonl")
# ====================================================

BASE_URL = "https://open.api.nexon.com/fconline/v1"
HEADERS = {"x-nxopen-api-key": API_KEY}

call_count = 0


class RateLimitedError(Exception):
    """429 등 레이트리밋/일일 한도 초과. 닉네임이 잘못된 것과는 다르므로 절대 버리지 않고
    큐에 되돌려놓은 뒤 실행을 즉시 멈춘다."""


def _get(path, params):
    """공통 GET 요청 래퍼. 호출 횟수를 세고, rate limit 간격을 지킨다."""
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


def get_match_list(ouid, matchtype, offset=0, limit=1):
    data = _get("/user/match", {
        "ouid": ouid,
        "matchtype": matchtype,
        "offset": offset,
        "limit": limit,
    })
    return data or []


def get_match_detail(match_id):
    return _get("/match-detail", {"matchid": match_id})


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        # seen_detail_match: 저장 여부와 무관하게 detail을 이미 조회한 matchId (큐 확장 중복 조회 방지)
        state.setdefault("seen_detail_match", list(state.get("seen_match", [])))
        # 기존 큐에 이미 들어있던 중복 닉네임 제거 (순서 유지)
        state["queue"] = list(dict.fromkeys(state.get("queue", [])))
        # queued_nicknames: 큐에 한 번이라도 들어간 적 있는 닉네임 전체 (재추가 방지)
        state.setdefault("queued_nicknames", list(state["queue"]))
        return state
    return {
        "queue": list(SEED_NICKNAMES),
        "seen_ouid": [],
        "seen_match": [],
        "seen_detail_match": [],
        "queued_nicknames": list(SEED_NICKNAMES),
    }


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def main():
    state = load_state()
    seen_ouid = set(state["seen_ouid"])
    seen_match = set(state["seen_match"])
    seen_detail_match = set(state["seen_detail_match"])
    queued_nicknames = set(state["queued_nicknames"])
    queue = state["queue"]

    collected_this_run = 0

    rate_limited = False

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "a", encoding="utf-8") as out:
        while queue and call_count < DAILY_CALL_BUDGET - 3:
            nickname = queue.pop(0)

            try:
                ouid = get_ouid(nickname)
                if not ouid or ouid in seen_ouid:
                    continue
                seen_ouid.add(ouid)

                # /user/match 응답은 matchId 문자열의 배열을 그대로 반환한다 (객체 배열이 아님)
                # 큐가 빨리 고갈되는 것을 완화하기 위해 최근 QUEUE_EXPAND_LIMIT경기까지 살펴보고
                # 상대방을 전부 큐에 넣는다. 단, matches.jsonl에 저장하는 건 "1유저 1경기" 원칙대로 1건만.
                match_list = get_match_list(ouid, MATCHTYPE, offset=0, limit=QUEUE_EXPAND_LIMIT)
                if not match_list:
                    continue

                saved_this_user = False
                for match_id in match_list:
                    if call_count >= DAILY_CALL_BUDGET - 3:
                        break
                    if not match_id or match_id in seen_detail_match:
                        continue

                    detail = get_match_detail(match_id)
                    seen_detail_match.add(match_id)
                    if not detail:
                        continue

                    # 큐 확장: 살펴본 경기 전부에서 상대방 닉네임을 뽑아 큐에 추가
                    # (이미 처리된 유저뿐 아니라, 아직 처리 안 했지만 이미 큐에 들어가 있는 닉네임도 걸러서
                    #  같은 사람이 큐에 중복으로 쌓이는 것을 막는다)
                    for info in detail.get("matchInfo", []):
                        opp_ouid = info.get("ouid")
                        opp_nick = info.get("nickname")
                        if opp_ouid and opp_ouid not in seen_ouid and opp_nick and opp_nick not in queued_nicknames:
                            queue.append(opp_nick)
                            queued_nicknames.add(opp_nick)

                    # 저장(학습 데이터)은 이 유저당 아직 저장 안 한 경기 중 첫 건만
                    if not saved_this_user and match_id not in seen_match:
                        seen_match.add(match_id)
                        out.write(json.dumps(detail, ensure_ascii=False) + "\n")
                        collected_this_run += 1
                        saved_this_user = True
            except RateLimitedError as e:
                # 레이트리밋/일일한도 초과는 "잘못된 닉네임"이 아니므로 큐에서 버리지 않고 되돌려놓는다
                queue.insert(0, nickname)
                print(f"\n  [중단] 레이트리밋 감지, 실행을 멈춘다: {e}")
                rate_limited = True
                break

            if collected_this_run and collected_this_run % 20 == 0:
                print(f"  진행: {collected_this_run}경기 수집, API 호출 {call_count}회, 큐 {len(queue)}명 대기")

    state["queue"] = queue
    state["seen_ouid"] = list(seen_ouid)
    state["seen_match"] = list(seen_match)
    state["seen_detail_match"] = list(seen_detail_match)
    state["queued_nicknames"] = list(queued_nicknames)
    save_state(state)

    print(f"\n=== 이번 실행 요약 ===")
    print(f"수집한 경기 수: {collected_this_run}")
    print(f"API 호출 횟수: {call_count}")
    print(f"누적 고유 유저 수: {len(seen_ouid)}")
    print(f"누적 고유 경기 수: {len(seen_match)}")
    print(f"다음 실행 대기 큐: {len(queue)}명")
    print(f"저장 파일: {OUT_FILE} (누적 append 방식)")
    if rate_limited:
        print("레이트리밋(429)으로 중단됨 — 일일 한도가 다 찼을 가능성이 높다. 시간을 두고 다시 실행할 것.")


if __name__ == "__main__":
    main()
