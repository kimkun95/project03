import html

import streamlit as st

from api import diagnose
from board import draw_board
from util import (
    error_stop,
    extract_style_summary,
    fit_status,
    get_weakest_slot,
    is_valid_number,
    player_name_for,
    to_percent,
    to_score,
    valid_squad_slots,
)


def show():
    safe_nickname = html.escape(st.session_state.nickname)

    st.markdown(
        f"""
        <div class="result-hero">
            <div class="result-hero-title">⚽ <span>{safe_nickname}</span> 감독님의 AI 전술 분석 리포트</div>
            <div class="result-hero-sub">최근 공식경기 데이터와 학습된 K-means 모델을 기반으로 분석한 결과입니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("⬅️ 다시 검색", use_container_width=True):
            st.session_state.page = "search"
            st.session_state.diagnosis_result = None
            st.session_state.diagnosis_nickname = None
            st.rerun()

    # -----------------------------------------------------
    # 실제 진단 실행
    # Streamlit rerun 때 API를 다시 호출하지 않도록 세션에 저장
    # -----------------------------------------------------
    # 아직 진단 결과가 없거나 다른 닉네임 결과라면 실제 백엔드 진단 실행
    if (
        st.session_state.diagnosis_result is None
        or st.session_state.diagnosis_nickname != st.session_state.nickname
    ):
        try:
            with st.spinner("최근 공식경기와 현재 스쿼드를 분석하고 있습니다..."):
                # ★ 핵심: 여기서 live_diagnosis.diagnose(닉네임)가 호출됨
                # 반환된 result 안에 플레이스타일, 궁합점수, 스쿼드 슬롯 등이 들어있음
                result = diagnose(st.session_state.nickname)
            st.session_state.diagnosis_result = result
            st.session_state.diagnosis_nickname = st.session_state.nickname
        except Exception as exc:
            error_text = str(exc)
            if "latin-1" in error_text or "codec can't encode" in error_text:
                guide = (
                    "NEXON_API_KEY가 정상적으로 로드되지 않았을 가능성이 큽니다. "
                    "app.py 옆의 .env 파일을 확인해 주세요."
                )
            else:
                guide = (
                    "NEXON_API_KEY, 모델 파일 경로, "
                    "player_stats_final.csv 경로를 확인해 주세요."
                )

            error_stop(
                f"진단 중 오류가 발생했습니다. {guide}\n\n"
                f"오류 내용: {error_text}"
            )

    result = st.session_state.diagnosis_result

    if not result:
        error_stop("진단 결과를 불러오지 못했습니다.")

    if result.get("error"):
        error_stop(f"진단할 수 없습니다: {result['error']}")

    # [13] 백엔드 result에서 화면별로 사용할 핵심 데이터 꺼내기
    # user_style   : 숏패스/스루패스/슈팅 등 사용자 플레이스타일 지표
    user_style = result.get("user_style") or {}
    squad_scores = result.get("squad_scores") or {}
    # valid_slots : position_fit 계산이 가능한 선수만 모은 목록
    valid_slots = valid_squad_slots(result)
    # weakest_slot: 현재 스쿼드에서 궁합이 가장 낮은 선수/포지션
    weakest_slot = get_weakest_slot(result)
    # squad_fit_score: 스쿼드 전체 스타일 궁합도(0~1)
    squad_fit_score = squad_scores.get("squad_fit_score")

    st.write("")

    # [14] 결과 화면을 3개 탭으로 분리
    tab1, tab2, tab3 = st.tabs([
        "📊 1. 플레이스타일 분석",
        "💡 2. 스쿼드 궁합 진단",
        "📋 3. 포지션별 궁합",
    ])

    # =====================================================
    # TAB 1: 실제 K-means 스타일 분석
    # =====================================================
    # TAB 1 = 사용자의 실제 경기 데이터를 이용한 플레이스타일 분석
    with tab1:
        st.subheader("팀 플레이스타일 분석")

        summary = html.escape(extract_style_summary(result))

        st.markdown(
            f"""
            <div class="info-card-styled">
                <b>💡 AI 플레이스타일 진단</b><br>
                <span style="color:#475569;">{summary}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("##### 📈 실제 경기 기반 스타일 지표")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "숏패스 비율",
            to_percent(user_style.get("short_pass_ratio")),
        )

        m2.metric(
            "스루패스 비율",
            to_percent(user_style.get("through_pass_ratio")),
        )

        m3.metric(
            "드리븐 패스 비율",
            to_percent(user_style.get("driven_ground_pass_ratio")),
        )

        m4, m5, m6 = st.columns(3)
        m4.metric(
            "박스 안 슈팅 비율",
            to_percent(user_style.get("in_penalty_shoot_ratio")),
        )
        m5.metric(
            "중거리 슈팅 비율",
            to_percent(user_style.get("out_penalty_shoot_ratio")),
        )
        m6.metric(
            "헤딩 슈팅 비율",
            to_percent(user_style.get("heading_shoot_ratio")),
        )

        dribble_intensity = user_style.get("dribble_intensity")
        if is_valid_number(dribble_intensity):
            st.caption(f"참고 지표 · 드리블 강도: {float(dribble_intensity):.2f}")

    # =====================================================
    # TAB 2: 실제 스쿼드 궁합 결과
    # =====================================================
    # TAB 2 = 현재 스쿼드가 사용자 플레이스타일과 얼마나 잘 맞는지 요약
    with tab2:
        st.subheader("현재 스쿼드 스타일 궁합")

        # 왼쪽: 전체 궁합도 / 오른쪽: 가장 약한 포지션 경고
        col_score, col_alert = st.columns([1, 2])

        with col_score:
            st.metric(
                label="스쿼드 스타일 궁합도",
                value=to_percent(squad_fit_score),
            )

        with col_alert:
            if weakest_slot:
                weakness_text = (
                    f"가장 낮은 포지션은 <b>{html.escape(str(weakest_slot['pos_name']))}</b>이며 "
                    f"현재 스타일 궁합도는 <b>{to_percent(weakest_slot['position_fit'])}</b>입니다."
                )
            else:
                weakness_text = "포지션별 궁합을 계산할 수 있는 카드 데이터가 없습니다."

            st.markdown(
                f"""
                <div class="highlight-box-danger">
                    <b>⚠️ 우선 확인할 포지션</b><br>{weakness_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        col_rec1, col_rec2 = st.columns(2)

        with col_rec1:
            st.markdown("#### 🎯 보완 우선순위")

            if valid_slots:
                lowest_slots = sorted(
                    valid_slots,
                    key=lambda slot: float(slot["position_fit"]),
                )[:3]

                items = "".join(
                    f"<li><b>{html.escape(str(slot['pos_name']))}</b> · "
                    f"{html.escape(player_name_for(slot.get('sp_id')))} · "
                    f"{to_percent(slot['position_fit'])}</li>"
                    for slot in lowest_slots
                )
            else:
                items = "<li>계산 가능한 포지션 데이터가 없습니다.</li>"

            st.markdown(
                f'<div class="info-card-styled custom-list"><ul>{items}</ul></div>',
                unsafe_allow_html=True,
            )

        with col_rec2:
            st.markdown("#### 📋 진단 가이드")

            tips = []
            if weakest_slot:
                tips.append(
                    f"{weakest_slot['pos_name']} 포지션부터 현재 플레이스타일에 맞는 카드 스탯을 비교해 보세요."
                )
            tips.append("궁합 점수는 승리 확률이 아니라 사용자 플레이스타일과 카드 스탯의 적합도입니다.")
            tips.append("수비수와 GK는 스타일 대응 축이 없어 포지션 궁합 계산 대상에서 제외될 수 있습니다.")

            tips_html = "".join(f"<li>{html.escape(tip)}</li>" for tip in tips)
            st.markdown(
                f'<div class="info-card-styled custom-list"><ul>{tips_html}</ul></div>',
                unsafe_allow_html=True,
            )

        match_result = result.get("squad_match_result")
        if match_result:
            st.caption(f"현재 스쿼드 추출에 사용된 최근 정상종료 경기 결과: {match_result}")

    # =====================================================
    # TAB 3: 실제 스쿼드 전술보드 + 선수/포지션 궁합 통합 분석
    # =====================================================
    # TAB 3 = 실제 스쿼드 전술보드 + 개별 선수 궁합 상세 분석
    with tab3:
        st.subheader("⚽ 실제 스쿼드 전술보드 & 선수별 궁합 분석")

        # all_slots: 최근 경기에서 사용한 실제 스쿼드 선수 목록
        all_slots = (squad_scores.get("slots") or [])
        if not all_slots:
            st.info("현재 스쿼드 정보를 불러올 수 없습니다.")
        else:
            # 분석 가능한 선수부터 보여주되, 수비/GK도 전술보드에는 그대로 포함한다.
            selectable_slots = valid_slots if valid_slots else all_slots
            option_map = {}
            for slot in selectable_slots:
                name = player_name_for(slot.get("sp_id"))
                pos = str(slot.get("pos_name", "-"))
                label = f"{pos} · {name}"
                # 혹시 같은 라벨이 겹치면 내부적으로 spId를 덧붙여 키 충돌만 방지
                if label in option_map:
                    label = f"{label} ({slot.get('sp_id')})"
                option_map[label] = slot

            default_index = 0
            if weakest_slot:
                weakest_id = weakest_slot.get("sp_id")
                labels = list(option_map.keys())
                for i, label in enumerate(labels):
                    if option_map[label].get("sp_id") == weakest_id:
                        default_index = i
                        break

            # 사용자가 상세 분석할 선수를 선택하는 드롭다운
            selected_label = st.selectbox(
                "📍 전술보드에서 확인할 선수",
                list(option_map.keys()),
                index=default_index,
                help="선수를 바꾸면 전술보드 강조와 오른쪽 상세 분석이 함께 변경됩니다.",
            )
            # 선택한 선수의 전체 데이터 / 이름 / 포지션을 꺼냄
            target_slot = option_map[selected_label]
            target_name = player_name_for(target_slot.get("sp_id"))
            target_pos = str(target_slot.get("pos_name", "-"))

            # 왼쪽 = 전술보드 / 오른쪽 = 선택 선수 상세 궁합
            board_col, analysis_col = st.columns([2.1, 1], gap="medium")

            with board_col:
                st.markdown("#### 🗺️ 최근 경기 실제 스쿼드")
                fig = draw_board(
                    all_slots,
                    selected_sp_id=target_slot.get("sp_id"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("최근 정상종료 경기의 실제 11명을 배치했습니다. 노란색 선수가 현재 상세 분석 대상입니다.")

            with analysis_col:
                st.markdown(
                    f"""
                    <div style="
                        display: flex;
                        align-items: center;
                        gap: 7px;
                        white-space: nowrap;
                        margin-bottom: 6px;
                    ">
                        <span style="font-size: 22px;">👤</span>
                        <span style="
                            font-size: 22px;
                            font-weight: 700;
                            line-height: 1.2;
                            color: #0f172a;
                        ">
                            {html.escape(target_name)}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(f"실제 포지션 · {target_pos}")

                if is_valid_number(target_slot.get("position_fit")):
                    st.metric(
                        label="종합 스타일 궁합",
                        value=to_score(target_slot.get("position_fit")),
                        delta=fit_status(target_slot.get("position_fit")),
                    )
                else:
                    st.metric(label="종합 스타일 궁합", value="계산 대상 아님")

                st.write("")
                m_pass, m_shoot = st.columns(2)
                with m_pass:
                    st.markdown(
                        f"""
                        <div class="mini-fit-card">
                            <div class="mini-fit-label">패스 궁합</div>
                            <div class="mini-fit-value">
                                {to_score(target_slot.get("pass_fit"))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with m_shoot:
                    st.markdown(
                        f"""
                        <div class="mini-fit-card">
                            <div class="mini-fit-label">슈팅 궁합</div>
                            <div class="mini-fit-value">
                                {to_score(target_slot.get("shoot_fit"))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                note = target_slot.get("note")
                if note:
                    detail_text = html.escape(str(note))
                elif is_valid_number(target_slot.get("position_fit")):
                    detail_text = "선수 카드 스탯과 감독님의 실제 플레이스타일 비율을 결합해 계산한 포지션 궁합입니다."
                else:
                    detail_text = "현재 모델은 수비수/GK에 대응하는 스타일 축이 없어 개인 궁합 점수를 계산하지 않습니다."

                st.markdown(
                    f"""
                    <div class="info-card-styled analysis-detail-card">
                        <b>📋 선수 분석 정보</b><br><br>
                        <span style="color:#334155;">
                            <b>선수:</b> {html.escape(target_name)}<br>
                            <b>포지션:</b> {html.escape(target_pos)}<br><br>
                            {detail_text}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 해당 선수가 약점 포지션이면 별도 안내 박스를 보여준다.
                if weakest_slot and target_slot.get("sp_id") == weakest_slot.get("sp_id"):
                    st.markdown(
                        f"""
                        <div class="analysis-warning-box">
                            현재 계산 가능한 선수 중 <b>{html.escape(target_name)}({html.escape(target_pos)})</b>의 궁합이 가장 낮습니다.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.write("")
            st.divider()
            st.markdown("#### 📊 현재 스쿼드 선수별 스타일 궁합")

            # 궁합 계산 가능한 선수는 높은 점수순,
            # 계산 대상이 아닌 수비수/GK는 뒤쪽에 표시
            ranked_slots = sorted(
                all_slots,
                key=lambda x: (
                    not is_valid_number(x.get("position_fit")),
                    -float(x["position_fit"]) if is_valid_number(x.get("position_fit")) else 0,
                ),
            )

            for slot in ranked_slots:
                name = player_name_for(slot.get("sp_id"))
                pos = str(slot.get("pos_name", "-"))
                fit = slot.get("position_fit")
                if is_valid_number(fit):
                    score = max(0.0, min(float(fit), 1.0))
                    st.write(f"**{pos} · {name}** · {to_percent(fit)}")
                    st.progress(score)
                else:
                    st.write(f"**{pos} · {name}** · 궁합 계산 대상 아님")
