import os

import streamlit as st

# Streamlit 기본 설정
st.set_page_config(
    page_title="스쿼드 궁합 진단 대시보드",
    page_icon="⚽",
    layout="centered",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSS 불러오기
css_path = os.path.join(BASE_DIR, "main.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )

# 세션 초기값
if "page" not in st.session_state:
    st.session_state.page = "search"

if "nickname" not in st.session_state:
    st.session_state.nickname = ""

if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None

if "diagnosis_nickname" not in st.session_state:
    st.session_state.diagnosis_nickname = None

# 화면 불러오기
from search import show as show_search
from result import show as show_result

if st.session_state.page == "search":
    show_search()

elif st.session_state.page == "result":
    show_result()
