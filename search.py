import streamlit as st


def show():
    """메인 검색 화면."""
    st.markdown(
        '<div class="main-title">⚽ AI 스쿼드 궁합 진단기</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    with st.form("search_form"):
        col1, col2 = st.columns([3.2, 1], gap="small")

        with col1:
            nickname = st.text_input(
                "감독 닉네임",
                value=st.session_state.nickname,
                placeholder="FC 온라인 감독 닉네임을 입력해 주세요",
                label_visibility="collapsed",
            )

        with col2:
            search_button = st.form_submit_button(
                "🔍 진단하기",
                use_container_width=True,
            )

    if search_button:
        nickname = nickname.strip()

        if not nickname:
            st.warning("감독 닉네임을 입력해 주세요.")
            return

        if st.session_state.diagnosis_nickname != nickname:
            st.session_state.diagnosis_result = None
            st.session_state.diagnosis_nickname = None

        st.session_state.nickname = nickname
        st.session_state.page = "result"
        st.rerun()
