"""
포지션별 스타일 궁합 점수(position_fit_score) 계산.

USER_STYLE_PROFILE(패스 4개 + 슛 2개 비율)과 PLAYER_CARD(player_stats_final.csv,
필드플레이어 스탯) 사이의 가중평균으로 "이 카드가 유저 스타일에 얼마나 맞는가"를
잰다. 설계 근거와 실험 과정은 Notion "포지션별 스타일 궁합 점수 설계 (2026-09-07)"
페이지에 정리되어 있다.

2026-09-08 변경: 원래는 코사인 유사도를 썼으나, 코사인 유사도는 벡터의 "방향(비율)"만
보고 "크기(실제 스탯 수준)"는 무시해서 필드 스탯이 전반적으로 낮은 골키퍼 카드가
비율만 우연히 맞으면 정상급 미드필더보다 높은 점수를 받는 문제가 실측(spid.json으로
실제 선수명 대조 + 실제 매치 데이터 기준 승률 상관관계 검증)으로 확인됐다. 스탯의
실제 크기가 점수에 반영되는 가중평균으로 교체했다 — _fit_score() docstring 참고.

핵심 원칙:
- 점수는 style_type(K-means 군집 라벨)이 아니라 원래의 연속값 비율
  (preprocess_style.py의 PASS_FEATURE_COLUMNS/HEADING_FEATURE_COLUMNS/
  SHOT_LOCATION_FEATURE_COLUMNS)로 계산한다.
  라벨은 진단 문장("당신은 스루패스 위주 스타일입니다") 전용이고, 점수는 경계선에
  걸친 유저도 부드럽게 나오도록 연속값을 그대로 쓴다.
- pass_fit_score와 shoot_fit_score는 처음부터 끝까지 완전히 독립적으로 계산한 뒤,
  포지션 성격에 따라 맨 마지막에만 합친다. 6개 feature를 한 공간에서 통합
  군집화했을 때 실루엣 점수가 떨어졌던 것과 같은 이유(서로 다른 축을 섞으면
  신호가 희석됨)로, 궁합 점수도 처음부터 섞지 않는다.
- 같은 스탯이 여러 축/포지션에 중복으로 들어가도 된다 — 각 축에 들어간 이유가
  그 축 고유의 이유이기만 하면 된다(다른 축의 이유를 빌려오면 안 됨).
- 좌/우 미러 포지션(RDM/LDM, RCM/LCM, RAM/LAM, RM/LM, RF/LF, RW/LW, RS/LS)은
  축구 역할이 완전히 같아 스탯 매핑도 동일하다.
- 스트라이커(RS/ST/LS)는 패스를 "보내는" 게 아니라 "받는" 입장이라, pass_fit의
  스탯 매핑이 다른 포지션과 근본적으로 다르다(볼 컨트롤/몸싸움/속력처럼 수신 관련
  스탯). 이름은 같은 pass_fit_score지만 의미가 다르다는 점에 주의.
- 수비(spPosition 1~8)와 GK(0)는 대응되는 스타일 축 자체가 없다고 보고 궁합 점수
  대상에서 완전히 제외한다.
- 축당 스탯을 2~3개로 제한한다. 스탯을 계속 추가하면 (1) 평균이 뭉개지고
  (2) 서로 상관관계 높은 FIFA 스탯 특성상 "스타일 궁합"이 아니라 그냥
  "종합 능력치"(avg_stat_score와 역할이 겹침)가 되어버린다.

2026-09-07 추가, 2026-09-08 갱신: shoot_fit은 preprocess_style.py가 군집화(K-means)에
쓰는 HEADING_FEATURE_COLUMNS/SHOT_LOCATION_FEATURE_COLUMNS(각각 1축씩 분리됨, 2026-09-08
결정)이 아니라, 여기서만 쓰는 3축(SHOOT_FIT_AXES = in_penalty/out_penalty/heading)을
쓴다. 궁합 점수에서는 "중거리슛" 카드 스탯을 매핑할 자리가 필요해서
out_penalty_shoot_ratio도 함께 쓴다 — 군집 라벨과 궁합 점수는 어차피 독립적으로
계산되므로(모듈 docstring 원칙 참고) 서로 다른 feature 집합을 써도 문제없다.

2026-09-07 추가: BOTH_GROUPS의 position_fit_score 결합은 원래 pass_fit_score/
shoot_fit_score를 무조건 50:50 평균했는데, CAM이 순수 스트라이커와 같은 비중으로
슛 궁합을 받는 게 축구 상식과 안 맞는다는 지적으로 역할별 가중치(BOTH_GROUP_WEIGHTS)를
도입했다. 자세한 근거는 그 상수 정의부 주석 참고.

2026-09-08 추가: BOTH_GROUPS에 DM/CM/WIDE_MID도 포함시켰다 — out_penalty_shoot_ratio
(중거리슛 비중) 축은 원래도 "포지션 성격과 무관하게 핵심"이라는 이유로 다른 4개
그룹에 동일 매핑돼 있었는데, 정작 중거리슛을 스트라이커 다음으로 자주 시도하는
DM/CM/WIDE_MID는 반영이 안 되고 있었다. SHOOT_STAT_MAP에 이 세 그룹은
out_penalty_shoot_ratio 축만 추가했다(박스 안 마무리/헤더는 본업이 아니라 매핑
자체가 없음).
"""

import os

import numpy as np
import pandas as pd

import preprocess_style as preprocess

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYER_STATS_FILE = os.path.join(REPO_ROOT, "data", "player_stats_final.csv")
PLAYER_ID_COL = "spid"

# _fit_score의 가중평균(raw 스탯값 기준) 결과를 "1.0 만점"에 가깝게 맞추기 위한
# 정규화 상수. FIFA/FC 계열 스탯의 전통적 표시 상한(99)을 기준으로 삼는다.
# player_stats_final.csv에는 강화(spGrade) 반영으로 99를 넘는 값도 있어(최대 132,
# data-schema.md 미해결 항목 참고) 극강화 카드는 궁합 점수가 1.0을 살짝 넘을 수 있다 —
# 강화단계 반영 여부가 불분명한 상태에서 임의로 클리핑하지 않고 그대로 둔다.
STAT_DISPLAY_MAX = 130

# --- 포지션 그룹 (api-constraints.md의 spPosition 28개 코드 기준) ---
GK_POSITION = 0
DEFENSE_POSITIONS = {1, 2, 3, 4, 5, 6, 7, 8}       # SW/RWB/RB/RCB/CB/LCB/LB/LWB
SUB_POSITION = 28

DM_POSITIONS = {9, 10, 11}             # RDM, CDM, LDM
CM_POSITIONS = {13, 14, 15}            # RCM, CM, LCM
WIDE_MID_POSITIONS = {12, 16}          # RM, LM
CAM_POSITIONS = {17, 18, 19}           # RAM, CAM, LAM
DEEP_FORWARD_POSITIONS = {20, 21, 22}  # RF, CF, LF
WIDE_FORWARD_POSITIONS = {23, 27}      # RW, LW
STRIKER_POSITIONS = {24, 25, 26}       # RS, ST, LS

# 포지션 그룹별 position_fit_score 결합 방식. 2026-09-08 이전에는 DM/CM/WIDE_MID를
# PASS_ONLY_GROUPS로 따로 둬서 shoot_fit_score를 아예 계산하지 않았다 — 그런데
# out_penalty_shoot_ratio(중거리슛 비중) 축은 "포지션 성격과 무관하게 중거리 슛
# 기술 자체가 핵심"이라는 이유로 이미 다른 4개 그룹에 전부 동일하게 매핑돼 있었고,
# 실제로 중거리슛은 CM/DM이 스트라이커 다음으로 자주 시도하는 플레이라 이 셋만
# 빼놓을 근거가 약했다(2026-09-08 팀 결정으로 반영). 그래서 이제 7개 그룹 전부
# BOTH_GROUPS다 — DM/CM/WIDE_MID는 SHOOT_STAT_MAP에 out_penalty_shoot_ratio 축만
# 있어서(박스 안 마무리력/헤더는 본업이 아니므로 매핑 자체가 없음), 유저의 헤딩/
# 박스안슛 성향은 이 세 그룹에 전혀 영향을 주지 않고 중거리슛 성향만 반영된다.
BOTH_GROUPS = {"DM", "CM", "WIDE_MID", "CAM", "DEEP_FORWARD", "WIDE_FORWARD", "STRIKER"}

# BOTH_GROUPS의 pass_fit_score/shoot_fit_score 결합 가중치 (pass_weight, shoot_weight),
# 2026-09-07 팀 결정(CAM/DEEP_FORWARD/WIDE_FORWARD/STRIKER), 2026-09-08 추가
# (DM/CM/WIDE_MID). 원래는 전부 50:50 평균이었는데, CAM이 순수 스트라이커와 같은
# 비중으로 슛 궁합을 받는 게 축구 상식과 안 맞는다는 지적으로 역할별 차등을 도입했다:
# CAM은 플레이메이킹이 본업이라 pass 비중을 높게, STRIKER는 마무리가 본업이라 shoot
# 비중을 높게 잡는다. DEEP_FORWARD(CF)는 연계와 마무리를 둘 다 맡는 롤이라 50:50을
# 유지하고, WIDE_FORWARD는 측면 크로스/서비스가 침투 마무리보다 살짝 우선이라고 봐서
# pass 쪽에 약간 무게를 둔다. DM/CM/WIDE_MID는 패스가 여전히 본업이고 중거리슛은
# 어디까지나 보조적인 옵션이라 shoot 비중을 낮게(0.15) 잡는다 — 세 그룹 다 같은
# 이유(패스가 본업)라 값도 동일하게 뒀다. 데이터로 학습된 값이 아니라 팀이 축구
# 지식으로 정한 값이며, avg_stat_score 같은 종합 능력치가 되지 않도록 각 그룹의
# 축(pass_fit_score, shoot_fit_score) 자체는 여전히 완전히 독립적으로 계산한 뒤
# 마지막에만 합친다.
BOTH_GROUP_WEIGHTS = {
    "DM": (0.85, 0.15),
    "CM": (0.85, 0.15),
    "WIDE_MID": (0.85, 0.15),
    "CAM": (0.7, 0.3),
    "DEEP_FORWARD": (0.5, 0.5),
    "WIDE_FORWARD": (0.6, 0.4),
    "STRIKER": (0.2, 0.8),
}

# shoot_fit은 K-means 군집화용 preprocess.SHOOT_FEATURE_COLUMNS(2축)이 아니라
# 여기서만 쓰는 3축을 쓴다 (모듈 docstring 2026-09-07 추가 참고).
SHOOT_FIT_AXES = ["in_penalty_shoot_ratio", "out_penalty_shoot_ratio", "heading_shoot_ratio"]


def group_of(sp_position):
    """spPosition 코드를 7개 궁합 점수 그룹 중 하나로 분류한다.

    수비/GK/SUB이거나 알 수 없는 코드면 None을 반환한다(궁합 점수 대상 아님).
    """
    if sp_position in DM_POSITIONS:
        return "DM"
    if sp_position in CM_POSITIONS:
        return "CM"
    if sp_position in WIDE_MID_POSITIONS:
        return "WIDE_MID"
    if sp_position in CAM_POSITIONS:
        return "CAM"
    if sp_position in DEEP_FORWARD_POSITIONS:
        return "DEEP_FORWARD"
    if sp_position in WIDE_FORWARD_POSITIONS:
        return "WIDE_FORWARD"
    if sp_position in STRIKER_POSITIONS:
        return "STRIKER"
    return None


# --- 패스 축 스탯 매핑 (그룹 -> 축 -> 카드 스탯 컬럼명 리스트) ---
# 스트라이커만 "송신"이 아니라 "수신" 기준 스탯을 쓴다 (모듈 docstring 참고).
PASS_STAT_MAP = {
    "DM": {
        "short_pass_ratio": ["짧은 패스", "볼 컨트롤"],
        "long_pass_ratio": ["긴 패스", "시야"],
        "through_pass_ratio": ["시야"],
        "driven_ground_pass_ratio": ["볼 컨트롤"],
    },
    "CM": {
        "short_pass_ratio": ["짧은 패스", "볼 컨트롤"],
        "long_pass_ratio": ["긴 패스", "시야"],
        "through_pass_ratio": ["시야"],
        "driven_ground_pass_ratio": ["볼 컨트롤"],
    },
    "WIDE_MID": {
        "short_pass_ratio": ["짧은 패스", "볼 컨트롤"],
        "long_pass_ratio": ["긴 패스", "크로스"],
        "through_pass_ratio": ["시야"],
        "driven_ground_pass_ratio": ["크로스", "볼 컨트롤"],
    },
    "CAM": {
        "short_pass_ratio": ["짧은 패스", "볼 컨트롤"],
        "long_pass_ratio": ["긴 패스", "시야"],
        # 2026-09-08 수정: "가속력"(수신형 스탯) 제거 — 모듈 docstring 원칙상 CAM은
        # DEEP_FORWARD/STRIKER와 달리 "창조자"(송신 기준)라, 시야 하나만 써야
        # DM/CM/WIDE_MID/WIDE_FORWARD와 일관된다. 가속력이 섞여있으면 순수
        # 플레이메이커(파브레가스/피를로 등)보다 드리블러형 만능선수가 부당하게
        # 고득점하는 걸 실측으로 확인함.
        "through_pass_ratio": ["시야"],
        "driven_ground_pass_ratio": ["볼 컨트롤"],
    },
    "DEEP_FORWARD": {
        # short_pass/driven_ground은 콤비네이션 플레이라 송신 기준 유지.
        # long_pass/through_pass는 CAM(창조자)과 달리 CF는 최전방에서 그 패스를
        # "받는" 쪽에 훨씬 가까워서 ST와 같은 수신 기준 스탯을 쓴다.
        "short_pass_ratio": ["짧은 패스", "볼 컨트롤"],
        "long_pass_ratio": ["볼 컨트롤", "몸싸움"],
        "through_pass_ratio": ["속력", "가속력"],
        "driven_ground_pass_ratio": ["볼 컨트롤"],
    },
    "WIDE_FORWARD": {
        "short_pass_ratio": ["짧은 패스", "볼 컨트롤"],
        "long_pass_ratio": ["긴 패스", "크로스"],
        "through_pass_ratio": ["시야"],
        "driven_ground_pass_ratio": ["크로스", "볼 컨트롤"],
    },
    "STRIKER": {
        # 패스를 보내는 게 아니라 받는 입장 — 수신 관련 스탯으로 매핑.
        "short_pass_ratio": ["볼 컨트롤"],
        "long_pass_ratio": ["볼 컨트롤", "몸싸움"],
        "through_pass_ratio": ["속력", "가속력"],
        "driven_ground_pass_ratio": ["볼 컨트롤", "반응 속도"],
    },
}

# --- 슛 축 스탯 매핑 ---
# out_penalty_shoot_ratio(중거리슛 비중)는 포지션 성격과 무관하게 "중거리 슛" 기술
# 자체가 핵심이라 7개 그룹 모두 동일하게 매핑한다 (다른 축처럼 역할별 차별화 안 함).
_OUT_PENALTY_STATS = ["중거리 슛", "슛 파워"]

SHOOT_STAT_MAP = {
    # DM/CM/WIDE_MID는 2026-09-08부터 shoot_fit을 쓰지만, out_penalty_shoot_ratio
    # (중거리슛) 축만 있다 — 박스 안 마무리(in_penalty)나 헤더는 이 포지션들의
    # 본업이 아니라 애초에 매핑할 스탯 자체가 없다고 보고 축을 만들지 않았다.
    # 유저가 헤딩형/박스안형으로 진단돼도 CM/DM/WIDE_MID 카드 점수에는 영향이
    # 없고, 오직 중거리슛 성향(out_penalty_shoot_ratio)만 반영된다.
    "DM": {
        "out_penalty_shoot_ratio": _OUT_PENALTY_STATS,
    },
    "CM": {
        "out_penalty_shoot_ratio": _OUT_PENALTY_STATS,
    },
    "WIDE_MID": {
        "out_penalty_shoot_ratio": _OUT_PENALTY_STATS,
    },
    "CAM": {
        "in_penalty_shoot_ratio": ["골 결정력", "위치 선정"],
        "out_penalty_shoot_ratio": _OUT_PENALTY_STATS,
        "heading_shoot_ratio": ["헤더"],
    },
    "DEEP_FORWARD": {
        "in_penalty_shoot_ratio": ["골 결정력", "위치 선정", "발리슛"],
        "out_penalty_shoot_ratio": _OUT_PENALTY_STATS,
        "heading_shoot_ratio": ["헤더"],
    },
    "WIDE_FORWARD": {
        "in_penalty_shoot_ratio": ["골 결정력", "위치 선정", "속력"],
        "out_penalty_shoot_ratio": _OUT_PENALTY_STATS,
        # 본인이 헤딩하는 게 아니라 크로스로 헤딩 기회를 만들어주는 역할.
        "heading_shoot_ratio": ["크로스", "속력"],
    },
    "STRIKER": {
        "in_penalty_shoot_ratio": ["골 결정력", "위치 선정", "반응 속도"],
        "out_penalty_shoot_ratio": _OUT_PENALTY_STATS,
        "heading_shoot_ratio": ["헤더", "점프", "몸싸움"],
    },
}


def load_player_stats(path=PLAYER_STATS_FILE):
    """player_stats_final.csv를 로드해 spid를 인덱스로 한 DataFrame을 반환한다."""
    df = pd.read_csv(path)
    return df.set_index(PLAYER_ID_COL)


def _invert_axis_stat_map(axis_stat_map):
    """{axis: [stat, ...]} -> {stat: [axis, ...]}로 뒤집는다.

    한 스탯이 여러 축에 걸쳐 있으면(예: 볼 컨트롤이 short_pass_ratio와
    driven_ground_pass_ratio 둘 다에 관련 있음), 그 스탯은 여러 축의 리스트를 갖는다.
    """
    stat_to_axes = {}
    for axis, stats in axis_stat_map.items():
        for stat in stats:
            stat_to_axes.setdefault(stat, []).append(axis)
    return stat_to_axes


def _cosine_similarity(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


# _fit_score의 스타일_배율이 최종 점수를 흔들 수 있는 폭. 0.3이면 스타일 방향이
# 완전히 어긋나도(cos=-1) 배율이 0.4, 완전히 일치해도(cos=1) 배율이 1.0에 그쳐서,
# 품질 차이가 큰 카드끼리는 스타일만으로 순위가 안 뒤집힌다. 스타일을 더/덜
# 중요하게 보고 싶으면 이 값을 조정한다(축구 지식으로 정한 값, 데이터로 학습된
# 값 아님 — BOTH_GROUP_WEIGHTS와 같은 성격).
STYLE_INFLUENCE = 0.4


def _fit_score(user_style, card_stats, axis_stat_map):
    """"품질(quality) x 스타일_배율(style_multiplier)"로 궁합 점수를 계산한다.

    2026-09-08 재설계 (세 번째 버전). 앞서 두 방식을 실측으로 검증하며 각각
    반대 방향으로 실패하는 걸 확인했다:
    - 1차: 코사인 유사도(방향만 봄, 크기 무시) — 필드 스탯이 전반적으로 낮은
      골키퍼 카드가 비율만 우연히 맞으면 정상급 미드필더보다 높은 CM 궁합
      점수를 받음 (spid.json 실제 선수명 대조로 확인).
    - 2차: 가중평균(크기만 봄, 방향 무시) — 대신 압도적으로 뛰어난 만능 카드가
      유저가 신경도 안 쓰는 스탯에서 벌어들인 우위로, 정작 유저가 원하는
      스탯에서 밀리는 특화 카드를 이겨버리는 경우가 생김(스탯 공유/가중치 배분
      방식을 sum/max/완전분리/raw합 등 여러 방법으로 바꿔봐도 해결 안 됨 —
      근본 원인이 가중치 배분이 아니라 "크기만 보는 단일 지표"라는 구조 자체
      였음).

    그래서 아예 두 지표로 분리한다:
    1) quality: 이 포지션에 관련된 스탯 전체의 단순 평균(÷STAT_DISPLAY_MAX).
       유저 스타일과 완전히 무관 — "이 카드가 절대적으로 얼마나 좋은가"만 잰다.
    2) style_cos: 카드 자신의 스탯 값을 "구성비"(합이 1이 되도록 정규화)로 바꿔서
       유저가 원하는 구성비와 코사인 유사도를 잰다. 카드의 전체적인 실력 크기는
       정규화 과정에서 지워지고 "이 카드가 상대적으로 어디에 강한지"라는
       방향성만 남는다.

    이 둘을 곱으로 합친다(덧셈이면 "스타일 만점"이 나쁜 품질을 상쇄해버릴 수
    있어 곱셈으로 설계). style_cos는 STYLE_INFLUENCE만큼만 최종 배율에
    반영되도록 [1-STYLE_INFLUENCE, 1.0] 범위로 눌러써서, 품질 차이가 큰 카드
    사이의 순위를 스타일만으로 뒤집을 수 없게 한다 — 그래야 형편없는 카드가
    비율만 맞아서 이기는 1차 문제도, 반대로 만능 카드가 특화 카드를 압도해버리는
    2차 문제도 동시에 어느 정도 완화된다(후자는 품질 차이가 실제로 존재하는
    한 완전히는 못 없앤다 — 대화 로그의 호날두 vs 라르센 사례 참고. 그 경우는
    포뮬러가 아니라 "정말로 호날두가 더 나은 카드"라는 결론으로 정리됨).

    스탯을 축별로 평균 내던 예전 방식(_card_axis_value)은, 두 축이 완전히 같은
    스탯 조합을 쓰면(예: short_pass_ratio와 driven_ground_pass_ratio가 둘 다
    [짧은 패스, 볼 컨트롤]) 두 축의 카드 값이 항상 똑같아져서 유저가 어느 쪽을
    선호하든 카드 점수를 구분 못 하는 문제가 있었다. 지금 방식은 스탯별로 "그
    스탯이 관련 있는 유저 축들의 비율값을 더한 것"을 목표 비중으로 삼아 이
    문제를 피한다(다만 두 축의 관련 스탯 목록이 100% 동일하면 여전히 구분
    못 하므로, 그 경우엔 스탯 목록 자체를 다르게 잡아야 한다 — PASS_STAT_MAP/
    SHOOT_STAT_MAP 주석 참고).
    """
    stat_to_axes = _invert_axis_stat_map(axis_stat_map)
    if not stat_to_axes:
        return np.nan

    stats = sorted(stat_to_axes)  # 순서 고정(목표 비중·카드 값이 같은 순서여야 함)
    weights = [sum(user_style[axis] for axis in stat_to_axes[stat]) for stat in stats]
    if sum(weights) == 0:
        return np.nan
    card_values = [card_stats[stat] for stat in stats]

    quality = float(np.mean(card_values)) / STAT_DISPLAY_MAX

    card_total = sum(card_values)
    if card_total == 0:
        style_cos = 0.0  # 스탯이 전부 0인 카드 — 방향성 자체가 정의 안 되므로 중립 취급
    else:
        target_profile = [w / sum(weights) for w in weights]
        card_profile = [v / card_total for v in card_values]
        style_cos = _cosine_similarity(target_profile, card_profile)

    style_multiplier = (1 - STYLE_INFLUENCE) + STYLE_INFLUENCE * style_cos
    return quality * style_multiplier


def compute_pass_fit_score(user_style, card_stats, sp_position):
    """유저의 패스 스타일 비율과 카드의 패스 관련 스탯 사이의 코사인 유사도.

    user_style: short_pass_ratio/long_pass_ratio/through_pass_ratio/
        driven_ground_pass_ratio 키를 가진 dict-like (예: user_style_profile.csv 한 행).
    card_stats: player_stats_final.csv 컬럼명을 키로 하는 dict-like (카드 한 장).
    sp_position: api-constraints.md 기준 spPosition 코드.

    해당 포지션 그룹이 pass_fit을 쓰지 않으면(GK/수비/스트라이커 외 매핑 없음) NaN.
    """
    group = group_of(sp_position)
    if group is None or group not in PASS_STAT_MAP:
        return np.nan
    return _fit_score(user_style, card_stats, PASS_STAT_MAP[group])


def compute_shoot_fit_score(user_style, card_stats, sp_position):
    """유저의 슛 스타일 비율과 카드의 슛 관련 스탯 사이의 코사인 유사도.

    user_style: in_penalty_shoot_ratio/out_penalty_shoot_ratio/heading_shoot_ratio
        키를 가진 dict-like (SHOOT_FIT_AXES 참고 — K-means 군집화용 2축이 아니라
        여기서만 쓰는 3축).
    해당 포지션 그룹이 shoot_fit을 쓰지 않으면(DM/CM/WIDE_MID/GK/수비) NaN.
    """
    group = group_of(sp_position)
    if group is None or group not in SHOOT_STAT_MAP:
        return np.nan
    return _fit_score(user_style, card_stats, SHOOT_STAT_MAP[group])


def _weighted_combine(pass_fit, shoot_fit, pass_weight, shoot_weight):
    """pass_fit/shoot_fit을 역할별 가중치로 합친다.

    카드 스탯이 전부 0이라 코사인 유사도 분모가 0이 되는 등, 드물게 둘 중 하나가
    NaN으로 나오는 경우에는 가중치를 무시하고 남은 값을 그대로 쓴다(전부 NaN이면
    NaN 유지) — 예전 np.nanmean 방식의 안전장치를 그대로 가져온 것.
    """
    if np.isnan(pass_fit) and np.isnan(shoot_fit):
        return np.nan
    if np.isnan(pass_fit):
        return shoot_fit
    if np.isnan(shoot_fit):
        return pass_fit
    return pass_weight * pass_fit + shoot_weight * shoot_fit


def compute_position_fit_score(user_style, card_stats, sp_position):
    """포지션 성격에 따라 pass_fit_score/shoot_fit_score를 BOTH_GROUP_WEIGHTS의
    역할별 가중치로 계산하고 결합한다 (7개 그룹 전부 동일한 방식, 2026-09-08부터
    DM/CM/WIDE_MID도 포함 — SHOOT_STAT_MAP에 out_penalty_shoot_ratio 축만 있어서
    사실상 "패스 위주 + 중거리슛 소폭 반영"이 된다).

    GK/수비/SUB/알 수 없는 코드는 NaN (궁합 점수 대상 아님).
    """
    group = group_of(sp_position)
    if group is None or group not in BOTH_GROUPS:
        return np.nan

    pass_fit = compute_pass_fit_score(user_style, card_stats, sp_position)
    shoot_fit = compute_shoot_fit_score(user_style, card_stats, sp_position)
    pass_weight, shoot_weight = BOTH_GROUP_WEIGHTS[group]
    return _weighted_combine(pass_fit, shoot_fit, pass_weight, shoot_weight)
