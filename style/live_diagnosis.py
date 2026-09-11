"""
실시간 스타일 진단 + 스쿼드 궁합 점수 ("진단하기" 버튼의 백엔드 로직, CLAUDE.md 핵심 기능 1~5번).

닉네임 -> ouid -> 최근 30경기 조회(2026-09-07 확정, CLAUDE.md 핵심 기능 2번 참고) ->
스타일 비율 계산 -> 학습된 K-means로 군집 배정 -> 가장 최근 "정상종료" 경기로 현재 스쿼드
자동구성(몰수 경기는 player[]가 비어있는 경우가 많아 제외, CLAUDE.md 핵심 기능 3번 참고) ->
position_fit.py로 포지션별/스쿼드 궁합 점수 -> 규칙 기반 진단 문장까지 한 번에 처리한다.

API 호출은 fetch_recent_matches()에서만 발생한다. 그 이후 단계(비율 계산, 스쿼드 구성,
궁합 점수 계산)는 전부 순수 함수라 네트워크 없이 테스트 가능하고, 대시보드에서 유저가
스쿼드 슬롯을 다른 카드로 바꿨을 때도 score_squad()만 다시 부르면 된다 — user_style은
그대로 재사용하고 API를 다시 호출할 필요가 없다("대표 스쿼드를 게임 내 지정값에서 가져오자"는
안은 기각됨 — Nexon 공식 Open API/공식 사이트 어디에도 그런 엔드포인트가 없고, 있는 건 로그인
필요한 웹 프로필 페이지뿐이라 크롤링+자동로그인은 계정 정지 위험이 있어 채택하지 않았다).
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

import position_fit as pf
import preprocess_style as preprocess

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "style_diagnosis_baseline.joblib")

API_KEY = os.environ.get("NEXON_API_KEY", "여기에_발급받은_API_키")
BASE_URL = "https://open.api.nexon.com/fconline/v1"
HEADERS = {"x-nxopen-api-key": API_KEY}
REQUEST_INTERVAL = 0.35  # winrate/snowball_collect.py와 동일 (초당 5건 제한 대응)

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

N_MATCHES = 20  # 2026-09-07 확정 (CLAUDE.md 핵심 기능 2번 참고)
MATCHTYPE = 50  # 공식경기

# ---------------------------------------------------------
# 병렬 API 호출용 Rate Limit 제어
# 여러 Thread가 동시에 요청하더라도
# 실제 요청 시작 시점은 REQUEST_INTERVAL만큼 간격을 둔다.
# ---------------------------------------------------------
_API_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_TIME = 0.0

SUB_POSITION = 28
POS_NAME = {
    0: "GK", 1: "SW", 2: "RWB", 3: "RB", 4: "RCB", 5: "CB", 6: "LCB", 7: "LB", 8: "LWB",
    9: "RDM", 10: "CDM", 11: "LDM", 12: "RM", 13: "RCM", 14: "CM", 15: "LCM", 16: "LM",
    17: "RAM", 18: "CAM", 19: "LAM", 20: "RF", 21: "CF", 22: "LF", 23: "RW", 24: "RS",
    25: "ST", 26: "LS", 27: "LW", 28: "SUB",
}

# 패스 모델: 군집 번호 -> 사후 라벨 (2026-09-08, 137명/MIN_MATCHES_PER_USER=50 기준
# data/style/style_diagnosis_baseline_summary.txt의 [군집별 feature 평균]을 보고, 각
# 군집이 다른 군집 대비 뚜렷하게 높은 feature를 기준으로 붙였다 —
# CLAUDE.md 원칙 #4: 이름은 군집 결과가 나온 뒤 사후에 붙인다).
#   0: short_pass_ratio가 압도적(0.763)
#   1: through_pass_ratio가 세 군집 중 가장 높음(0.201 vs 0.134/0.145)
#   2: driven_ground_pass_ratio가 세 군집 중 가장 높음(0.168 vs 0.051/0.058)
# ⚠️ models/style_diagnosis_baseline.joblib을 train_style.py로 다시 학습하면 군집 번호
# 순서가 바뀔 수 있다 — 재학습 시 반드시 새 요약 파일을 다시 보고 이 매핑을 갱신할 것.
# 실제로 2026-09-08 재학습 때 이 갱신을 빠뜨려 0번/1번이 뒤바뀐 채로 남아있던 걸 뒤늦게
# 발견해 바로잡았다 (패스는 feature가 4개라 헤딩/슛위치축처럼 "값 순서로 자동 정렬"이
# 안 통해 계속 하드코딩한다 — 그래서 특히 재학습 후 갱신을 잊지 않도록 주의할 것).
PASS_CLUSTER_LABELS = {
    0: "숏패스 위주",
    1: "스루패스 위주",
    2: "드리븐그라운드패스 위주",
}

# 헤딩축/슛위치축: feature가 1개뿐이라 "군집 중심값이 낮은 쪽 -> 높은 쪽" 순서로 아래
# 라벨을 자동 배정한다(_ordered_cluster_labels 참고). PASS_CLUSTER_LABELS와 달리 재학습
# 으로 군집 번호(0/1/2...)가 바뀌어도 라벨이 저절로 따라가므로 수동 갱신이 필요 없다 —
# 2026-09-08에 재학습 후 SHOOT_CLUSTER_LABELS를 안 고쳐서 라벨이 실제 군집 특성과
# 어긋났던 문제(3군집 모델 기준 라벨이 2군집 모델에 그대로 남아있었음)를 겪은 뒤 이
# 방식으로 바꿨다. k가 2가 아니게 재학습되면(현재까지 모든 실험에서 k=2가 최적이었음)
# 아래 리스트 길이와 안 맞아 자동으로 "군집 N" 형태 대체 라벨로 넘어간다.
HEADING_LABELS_ASCENDING = ["발슛 위주(헤딩 적은 편)", "헤딩슛을 섞어 쓰는 편"]
SHOT_LOCATION_LABELS_ASCENDING = ["박스 안(인패널티) 위주", "중거리슛을 섞어 쓰는 편"]


def _ordered_cluster_labels(kmeans, label_texts_ascending):
    """kmeans(단일 feature 기준 cluster_centers_)를 값이 낮은 순서로 정렬해, 그 순서대로
    label_texts_ascending을 군집 번호에 배정한다.

    표준화(StandardScaler)된 공간의 중심값이라도 원래 값과 대소 순서는 그대로 보존되므로
    (선형 변환, 스케일 계수가 항상 양수) 순위 매기기에 그대로 써도 된다. 군집 개수가
    label_texts_ascending 길이와 다르면(예: 재학습으로 k가 바뀌면) 빈 dict를 반환해
    호출부가 "군집 N" 형태의 대체 라벨로 자연스럽게 넘어가게 한다.
    """
    n_clusters = kmeans.cluster_centers_.shape[0]
    if n_clusters != len(label_texts_ascending):
        return {}
    order = np.argsort(kmeans.cluster_centers_[:, 0])
    return {int(cluster_id): label_texts_ascending[rank] for rank, cluster_id in enumerate(order)}
# ====================================================


class RateLimitedError(Exception):
    """429(rate limit) 응답. winrate/snowball_collect.py와 동일 의미."""


def _get(path, params):
    """
    Nexon Open API 공통 GET 요청 함수.

    병렬 호출을 사용하더라도 요청이 한꺼번에 몰리지 않도록
    각 요청 시작 사이에 REQUEST_INTERVAL만큼 간격을 둔다.

    기존 방식:
        요청 → 응답 기다림 → 0.35초 대기 → 다음 요청

    변경 방식:
        요청 시작 간격만 0.35초 유지하면서
        이전 요청의 응답을 기다리는 동안 다음 Thread가 준비될 수 있음
    """

    global _LAST_REQUEST_TIME

    # -----------------------------------------------------
    # API 요청 시작 시점 제어
    # -----------------------------------------------------
    with _API_REQUEST_LOCK:

        now = time.monotonic()

        elapsed = now - _LAST_REQUEST_TIME

        wait_time = REQUEST_INTERVAL - elapsed

        if wait_time > 0:
            time.sleep(wait_time)

        # 이번 요청의 시작 시간 기록
        _LAST_REQUEST_TIME = time.monotonic()

    # -----------------------------------------------------
    # 실제 API 요청
    # Lock 밖에서 실행하므로 여러 요청의 응답 대기는 병렬 처리됨
    # -----------------------------------------------------
    resp = SESSION.get(
        f"{BASE_URL}{path}",
        params=params,
        timeout=10
    )

    # -----------------------------------------------------
    # Rate Limit
    # -----------------------------------------------------
    if resp.status_code == 429:
        raise RateLimitedError(
            f"{path} 429: {resp.text[:200]}"
        )

    resp.raise_for_status()

    return resp.json()

def _fetch_single_match(match_id, ouid):
    """
    경기 1개의 상세 정보를 조회하고
    해당 감독(ouid)의 matchInfo만 추출한다.

    병렬 호출에서 각 Thread가 실행할 함수.
    """

    detail = _get(
        "/match-detail",
        {
            "matchid": match_id
        }
    )

    # 해당 경기에서 현재 검색한 감독의 데이터 찾기
    match_info = next(
        (
            m
            for m in detail.get("matchInfo", [])
            if m.get("ouid") == ouid
        ),
        None,
    )

    return match_info


def fetch_recent_matches(nickname, n_matches=N_MATCHES, matchtype=MATCHTYPE):
    """닉네임 -> ouid -> 최근 n_matches경기 상세 조회.

    API 호출이 발생하는 유일한 함수 (1 + 1 + n_matches회). 반환값은
    (ouid, [이 유저 기준 matchInfo dict, ...]) — 최신순 그대로.
    """
    ouid = _get("/id", {"nickname": nickname})["ouid"]

    match_ids = _get(
        "/user/match",
        {"ouid": ouid, "matchtype": matchtype, "offset": 0, "limit": n_matches},
    )

    # ---------------------------------------------------------
    # 경기 상세 병렬 조회
    # ---------------------------------------------------------
    match_infos = []

    # 동시에 최대 5개의 작업을 처리
    MAX_WORKERS = 5

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        # -----------------------------------------------------
        # match_ids 순서를 그대로 유지하면서 병렬 처리
        #
        # executor.map()은 병렬로 실행하지만
        # 결과 반환 순서는 입력한 match_ids 순서와 동일하다.
        #
        # 따라서 build_current_squad()에서
        # "가장 최근 정상종료 경기"를 찾는 기존 로직도 유지된다.
        # -----------------------------------------------------
        results = executor.map(
            lambda match_id: _fetch_single_match(
                match_id,
                ouid
            ),
            match_ids,
        )

    for match_info in results:

        # 혹시 해당 ouid의 matchInfo가 없는 비정상 데이터면 제외
        if match_info is not None:
            match_infos.append(match_info)


    return ouid, match_infos


def matches_to_style_rows(ouid, match_infos):
    """raw matchInfo 리스트 -> preprocess_style.extract_match_style_rows()가 기대하는
    컬럼의 DataFrame으로 변환한다 (비율 계산 로직은 중복 구현하지 않고 그대로 재사용하기
    위한 어댑터일 뿐).
    """
    rows = []
    for mi in match_infos:
        p, s = mi["pass"], mi["shoot"]
        rows.append({
            preprocess.MATCH_ID_COL: mi.get("matchId", ""),
            preprocess.OUID_COL: ouid,
            preprocess.MATCH_END_TYPE_COL: mi["matchDetail"]["matchEndType"],
            preprocess.PASS_TRY_COL: p.get("passTry", 0),
            preprocess.SHORT_PASS_TRY_COL: p.get("shortPassTry", 0),
            preprocess.LONG_PASS_TRY_COL: p.get("longPassTry", 0),
            "throughPassTry": p.get("throughPassTry", 0),
            "lobbedThroughPassTry": p.get("lobbedThroughPassTry", 0),
            preprocess.DRIVEN_GROUND_PASS_TRY_COL: p.get("drivenGroundPassTry", 0),
            preprocess.DRIBBLE_YARD_COL: mi["matchDetail"].get("dribble", 0),
            preprocess.SHOOT_TOTAL_COL: s.get("shootTotal", 0),
            preprocess.SHOOT_HEADING_COL: s.get("shootHeading", 0),
            preprocess.SHOOT_IN_PENALTY_COL: s.get("shootInPenalty", 0),
            preprocess.SHOOT_OUT_PENALTY_COL: s.get("shootOutPenalty", 0),
        })
    return pd.DataFrame(rows)


def compute_user_style(raw_rows_df):
    """matches_to_style_rows()가 만든 원시(camelCase) DataFrame -> position_fit.py가
    요구하는 키를 가진 유저 스타일 비율 dict.

    preprocess_style.extract_match_style_rows()로 몰수/오류 경기를 거르고 snake_case
    컬럼으로 바꾼 뒤, aggregate_user_style()로 비율을 낸다 — 두 단계 다 재사용(중복 구현
    금지). 라이브 진단은 학습용 70/50경기 하한선(MIN_MATCHES_PER_USER)과 무관하므로
    min_matches=1로 호출한다 — 유저 표본이 몇 경기든 있는 그대로 비율을 낸다.
    """
    rows_df = preprocess.extract_match_style_rows(raw_rows_df)
    style_df = preprocess.aggregate_user_style(rows_df, min_matches=1)
    if style_df.empty:
        return None
    row = style_df.iloc[0]
    columns = (
        preprocess.PASS_FEATURE_COLUMNS + preprocess.HEADING_FEATURE_COLUMNS
        + preprocess.SHOT_LOCATION_FEATURE_COLUMNS + preprocess.REFERENCE_COLUMNS
    )
    return {col: row[col] for col in columns}


def load_style_model(path=MODEL_PATH):
    """style_diagnosis_baseline.joblib을 로드한다 (모델별 scaler+kmeans+pca dict).

    diagnose()가 한 번만 로드해서 predict_style_clusters()와
    generate_diagnosis_sentence() 둘 다에 넘겨쓴다(파일을 두 번 읽지 않기 위함).
    """
    return joblib.load(path)


def predict_style_clusters(user_style, model=None):
    """학습된 모델로 pass/heading/shot_location 군집 라벨을 예측한다.

    model을 안 넘기면 MODEL_PATH에서 새로 로드한다(단독 호출/테스트 편의용).
    """
    if model is None:
        model = load_style_model()

    model_feature_columns = {
        "pass": preprocess.PASS_FEATURE_COLUMNS,
        "heading": preprocess.HEADING_FEATURE_COLUMNS,
        "shot_location": preprocess.SHOT_LOCATION_FEATURE_COLUMNS,
    }
    clusters = {}
    for key, feature_columns in model_feature_columns.items():
        x = [[user_style[c] for c in feature_columns]]
        x_scaled = model[key]["scaler"].transform(x)
        clusters[key] = int(model[key]["kmeans"].predict(x_scaled)[0])
    return clusters["pass"], clusters["heading"], clusters["shot_location"]


def build_current_squad(match_infos):
    """match_infos(최신순)에서 몰수가 아니고 11명이 온전히 채워진 가장 최근 경기를 찾아
    [{"sp_id": int, "sp_position": int}, ...] 11개를 반환한다.

    matchEndType==0만으로는 안전하지 않다 — 정상종료여도 드물게 player 수가 이상한
    레코드가 있을 수 있어, "SUB 제외 11명"까지 같이 확인한다. 못 찾으면 (None, 사유) 반환.
    """
    for mi in match_infos:
        if mi["matchDetail"]["matchEndType"] != 0:
            continue
        starters = [p for p in mi.get("player", []) if p["spPosition"] != SUB_POSITION]
        if len(starters) != 11:
            continue
        squad = [{"sp_id": p["spId"], "sp_position": p["spPosition"]} for p in starters]
        return squad, mi["matchDetail"].get("matchResult")

    return None, "정상종료 + 11명이 온전한 경기를 찾지 못했습니다"


def score_squad(user_style, squad, player_stats=None):
    """squad([{"sp_id", "sp_position"}, ...])의 포지션별/스쿼드 전체 궁합 점수.

    순수 함수 — API 호출 없음. 대시보드에서 유저가 슬롯 하나를 다른 카드로 바꾸면,
    바뀐 squad로 이 함수만 다시 부르면 된다(user_style은 재사용, 재계산 불필요).
    """
    if player_stats is None:
        player_stats = pf.load_player_stats()

    results = []
    for slot in squad:
        sp_id, sp_position = slot["sp_id"], slot["sp_position"]
        pos_name = POS_NAME.get(sp_position, str(sp_position))
        if sp_id not in player_stats.index:
            results.append({
                "sp_id": sp_id, "sp_position": sp_position, "pos_name": pos_name,
                "pass_fit": np.nan, "shoot_fit": np.nan, "position_fit": np.nan,
                "note": "카드 스탯 없음",
            })
            continue
        card = player_stats.loc[sp_id]
        pass_fit = pf.compute_pass_fit_score(user_style, card, sp_position)
        shoot_fit = pf.compute_shoot_fit_score(user_style, card, sp_position)
        position_fit = pf.compute_position_fit_score(user_style, card, sp_position)
        results.append({
            "sp_id": sp_id, "sp_position": sp_position, "pos_name": pos_name,
            "pass_fit": pass_fit, "shoot_fit": shoot_fit, "position_fit": position_fit,
            "note": None,
        })

    valid = [r for r in results if r["position_fit"] == r["position_fit"]]  # NaN 제외
    squad_fit_score = float(np.mean([r["position_fit"] for r in valid])) if valid else np.nan

    return {"slots": results, "squad_fit_score": squad_fit_score}


def generate_diagnosis_sentence(user_style, squad_scores, pass_cluster, heading_cluster,
                                 shot_location_cluster, model=None):
    """규칙 기반 진단 문장 (CLAUDE.md 핵심 기능 5번 예시 형식).

    특정 카드를 콕 집어 추천하지 않고, 가장 궁합이 낮은 포지션과 방향성만 제시한다.

    패스/슛 스타일은 유저 본인의 raw 비율 중 max()로 뽑지 않는다 — 축구 게임 특성상
    거의 모든 유저가 short_pass_ratio가 제일 커서, max()로는 사실상 항상 "숏패스 위주"만
    나오고 predict_style_clusters()가 계산한 K-means 군집 배정 결과가 버려지는 문제가
    있었다. 대신 그 유저가 실제로 배정된 pass_cluster/heading_cluster/shot_location_cluster
    번호를 라벨로 사후 매핑해 사용한다. 헤딩축/슛위치축은 PASS_CLUSTER_LABELS처럼
    하드코딩하지 않고 _ordered_cluster_labels()로 군집 중심값 순서에서 자동으로 뽑는다
    (재학습으로 군집 번호가 바뀌어도 라벨이 안 어긋나게 하기 위함 — 위 상수 정의부 참고).
    """
    if model is None:
        model = load_style_model()

    pass_label = PASS_CLUSTER_LABELS.get(pass_cluster, f"패스 군집 {pass_cluster}")
    heading_labels = _ordered_cluster_labels(model["heading"]["kmeans"], HEADING_LABELS_ASCENDING)
    heading_label = heading_labels.get(heading_cluster, f"헤딩 군집 {heading_cluster}")
    shot_location_labels = _ordered_cluster_labels(
        model["shot_location"]["kmeans"], SHOT_LOCATION_LABELS_ASCENDING
    )
    shot_location_label = shot_location_labels.get(
        shot_location_cluster, f"슛위치 군집 {shot_location_cluster}"
    )

    valid = [r for r in squad_scores["slots"] if r["position_fit"] == r["position_fit"]]
    lines = [f"당신은 {pass_label} · {shot_location_label} · {heading_label} 스타일입니다."]
    if valid:
        worst = min(valid, key=lambda r: r["position_fit"])
        lines.append(
            f"현재 스쿼드 궁합 점수는 {squad_scores['squad_fit_score']:.2f}(1.0 만점)이며, "
            f"{worst['pos_name']} 포지션의 궁합이 가장 낮습니다({worst['position_fit']:.2f}) — "
            f"이 자리에 맞는 스탯을 가진 카드를 고려해보세요."
        )
    return " ".join(lines)


def diagnose(nickname, n_matches=N_MATCHES):
    """최상위 오케스트레이터 — 대시보드는 이 함수 하나만 호출하면 된다."""
    ouid, match_infos = fetch_recent_matches(nickname, n_matches=n_matches)

    rows_df = matches_to_style_rows(ouid, match_infos)
    user_style = compute_user_style(rows_df)
    if user_style is None:
        return {"ouid": ouid, "error": "정상종료 경기가 없어 스타일을 계산할 수 없습니다"}

    model = load_style_model()
    pass_cluster, heading_cluster, shot_location_cluster = predict_style_clusters(user_style, model)

    squad, match_result = build_current_squad(match_infos)
    if squad is None:
        return {
            "ouid": ouid, "user_style": user_style,
            "pass_cluster": pass_cluster, "heading_cluster": heading_cluster,
            "shot_location_cluster": shot_location_cluster,
            "squad": None, "squad_scores": None, "sentence": None,
            "error": match_result,
        }

    squad_scores = score_squad(user_style, squad)
    sentence = generate_diagnosis_sentence(
        user_style, squad_scores, pass_cluster, heading_cluster, shot_location_cluster, model
    )

    return {
        "ouid": ouid,
        "user_style": user_style,
        "pass_cluster": pass_cluster,
        "heading_cluster": heading_cluster,
        "shot_location_cluster": shot_location_cluster,
        "squad": squad,
        "squad_match_result": match_result,
        "squad_scores": squad_scores,
        "sentence": sentence,
    }
