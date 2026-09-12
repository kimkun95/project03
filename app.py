import os
import base64

import streamlit as st


# ==========================================
# Streamlit 기본 설정
# ==========================================
st.set_page_config(
    page_title="스쿼드 궁합 진단 대시보드",
    page_icon="⚽",
    layout="centered",
)


# ==========================================
# 기본 경로
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

css_path = os.path.join(BASE_DIR, "main.css")
bg_image_path = os.path.join(BASE_DIR, "img", "1.png")


# ==========================================
# 이미지 Base64 변환 함수
# ==========================================
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


# ==========================================
# CSS 불러오기
# ==========================================
css = ""

if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()


# ==========================================
# 배경 이미지 적용
# ==========================================
if os.path.exists(bg_image_path):

    bg_image = get_base64_image(bg_image_path)

    # main.css 안의 __BACKGROUND_IMAGE__ 부분을
    # 실제 Base64 이미지로 교체
    css = css.replace(
        "__BACKGROUND_IMAGE__",
        f"data:image/png;base64,{bg_image}"
    )


# 최종 CSS 적용
st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True,
)


# ==========================================
# 세션 초기값
# ==========================================
if "page" not in st.session_state:
    st.session_state.page = "search"

if "nickname" not in st.session_state:
    st.session_state.nickname = ""

if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None

if "diagnosis_nickname" not in st.session_state:
    st.session_state.diagnosis_nickname = None


# ==========================================
# 화면 불러오기
# ==========================================
from search import show as show_search
from result import show as show_result


# ==========================================
# 페이지 분기
# ==========================================
if st.session_state.page == "search":
    show_search()

elif st.session_state.page == "result":
    show_result()