import os
import sys

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(
    dotenv_path=os.path.join(BASE_DIR, ".env"),
    override=True,
)

API_KEY = (os.getenv("NEXON_API_KEY") or "").strip()

if not API_KEY:
    raise RuntimeError(
        "NEXON_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인해 주세요."
    )

# live_diagnosis.py가 style 폴더에 있는 경우도 지원
STYLE_DIR = os.path.join(BASE_DIR, "style")

if os.path.isdir(STYLE_DIR) and STYLE_DIR not in sys.path:
    sys.path.insert(0, STYLE_DIR)

import live_diagnosis

live_diagnosis.API_KEY = API_KEY
live_diagnosis.HEADERS = {"x-nxopen-api-key": API_KEY}


@st.cache_data(ttl=300, show_spinner=False)
def diagnose(nickname):
    """같은 닉네임의 진단 결과를 5분간 캐시."""
    return live_diagnosis.diagnose(nickname)
