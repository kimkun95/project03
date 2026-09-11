import json
import math
import os

import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PASS_STYLE_LABELS = {
    0: "숏패스 위주",
    1: "스루패스 위주",
    2: "드리븐그라운드패스 위주",
}


def is_valid_number(value):
    """None / NaN / inf가 아닌 실제 숫자인지 확인."""
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def to_percent(value, digits=1):
    """0~1 비율값을 퍼센트 문자열로 변환."""
    if not is_valid_number(value):
        return "측정 불가"
    return f"{float(value) * 100:.{digits}f}%"


def to_score(value):
    """0~1 궁합값을 100점 기준 문자열로 변환."""
    if not is_valid_number(value):
        return "측정 불가"
    return f"{float(value) * 100:.1f}점"


def fit_status(value):
    """궁합 점수에 따라 상태 문구를 반환."""
    if not is_valid_number(value):
        return "데이터 없음"

    score = float(value) * 100
    if score >= 85:
        return "BEST 🔥"
    if score >= 75:
        return "GOOD 👍"
    if score >= 65:
        return "WARNING ⚠️"
    return "CHECK 🚨"


def valid_squad_slots(result):
    """궁합 계산이 가능한 선수만 반환."""
    squad_scores = result.get("squad_scores") or {}
    slots = squad_scores.get("slots") or []
    return [slot for slot in slots if is_valid_number(slot.get("position_fit"))]


def get_weakest_slot(result):
    """궁합 점수가 가장 낮은 선수를 반환."""
    slots = valid_squad_slots(result)
    if not slots:
        return None
    return min(slots, key=lambda slot: float(slot["position_fit"]))


def extract_style_summary(result):
    """진단 문장에서 핵심 스타일 설명만 추출."""
    sentence = result.get("sentence") or ""
    if not sentence:
        return "스타일 진단 결과가 없습니다."

    first_sentence = sentence.split(".", 1)[0].strip()
    if first_sentence.startswith("당신은 "):
        first_sentence = first_sentence[len("당신은 "):]
    return first_sentence


@st.cache_data(show_spinner=False)
def load_player_name_map():
    """spid.json을 읽어 선수 ID -> 선수명 형태로 반환."""
    candidates = [
        os.path.join(BASE_DIR, "spid.json"),
        os.path.join(BASE_DIR, "data", "spid.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            return {
                int(row["id"]): row["name"]
                for row in rows
                if "id" in row and "name" in row
            }

    return {}


PLAYER_NAME_MAP = load_player_name_map()


def player_name_for(sp_id):
    """선수 ID를 실제 선수명으로 변경."""
    if sp_id is None:
        return "선수 정보 없음"

    name = PLAYER_NAME_MAP.get(int(sp_id))
    return name if name else f"알 수 없는 선수 ({sp_id})"


def error_stop(message):
    """오류 메시지를 보여주고 검색 화면으로 돌아갈 수 있게 처리."""
    st.error(message)

    if st.button("⬅️ 다시 검색", use_container_width=False):
        st.session_state.page = "search"
        st.session_state.diagnosis_result = None
        st.session_state.diagnosis_nickname = None
        st.rerun()

    st.stop()
