import html

import plotly.graph_objects as go

from util import player_name_for, to_percent


POSITION = {
    "GK": (50, 3), "SW": (50, 17),

    "RWB": (88, 34), "RB": (84, 25), "RCB": (63, 25), "CB": (50, 23),
    "LCB": (37, 25), "LB": (16, 25), "LWB": (12, 34),

    "RDM": (63, 44), "CDM": (50, 42), "LDM": (37, 44),

    "RM": (84, 59), "RCM": (63, 57), "CM": (50, 56),
    "LCM": (37, 57), "LM": (16, 59),

    "RAM": (75, 72), "CAM": (50, 75), "LAM": (25, 72),

    "RF": (65, 86), "CF": (50, 85), "LF": (35, 86),

    "RW": (82, 95), "RS": (63, 95), "ST": (50, 95),
    "LS": (37, 95), "LW": (18, 95),
}


def draw_board(slots, selected_sp_id=None):
    """최근 경기 실제 스쿼드를 전술보드에 표시."""
    fig = go.Figure()

    field_left, field_right, field_bottom, field_top = 6, 94, -5, 120
    center_y = (field_bottom + field_top) / 2

    fig.add_shape(
        type="rect",
        x0=field_left, y0=field_bottom,
        x1=field_right, y1=field_top,
        line=dict(color="#ffffff", width=2.2),
        layer="below",
    )

    fig.add_shape(
        type="line",
        x0=field_left, y0=center_y,
        x1=field_right, y1=center_y,
        line=dict(color="#ffffff", width=2),
        layer="below",
    )

    fig.add_shape(
        type="circle",
        x0=42, y0=center_y - 8,
        x1=58, y1=center_y + 8,
        line=dict(color="#ffffff", width=2),
        layer="below",
    )

    fig.add_shape(
        type="rect",
        x0=24, y0=field_bottom,
        x1=76, y1=16,
        line=dict(color="#ffffff", width=2),
        layer="below",
    )

    fig.add_shape(
        type="rect",
        x0=39, y0=field_bottom,
        x1=61, y1=6,
        line=dict(color="#ffffff", width=2),
        layer="below",
    )

    fig.add_shape(
        type="rect",
        x0=24, y0=99,
        x1=76, y1=field_top,
        line=dict(color="#ffffff", width=2),
        layer="below",
    )

    fig.add_shape(
        type="rect",
        x0=39, y0=109,
        x1=61, y1=field_top,
        line=dict(color="#ffffff", width=2),
        layer="below",
    )

    coord_counts = {}

    for slot in slots:
        pos = str(slot.get("pos_name", ""))
        x, y = POSITION.get(pos, (50, 56))

        key = (x, y)
        count = coord_counts.get(key, 0)
        coord_counts[key] = count + 1

        if count:
            offset = 4.5 * count
            x += offset if count % 2 == 1 else -offset

        sp_id = slot.get("sp_id")
        name = player_name_for(sp_id)
        selected = str(sp_id) == str(selected_sp_id)

        marker_color = "#f59e0b" if selected else "#10b981"
        marker_size = 40 if selected else 31

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(
                    size=marker_size,
                    color=marker_color,
                    line=dict(
                        width=3 if selected else 2,
                        color="#ffffff",
                    ),
                ),
                text=[f"<b>{pos}</b>"],
                textposition="middle center",
                textfont=dict(color="white", size=9),
                hovertemplate=(
                    f"<b>{html.escape(name)}</b><br>"
                    f"포지션: {html.escape(pos)}<br>"
                    f"종합 궁합: {to_percent(slot.get('position_fit'))}<br>"
                    f"패스 궁합: {to_percent(slot.get('pass_fit'))}<br>"
                    f"슈팅 궁합: {to_percent(slot.get('shoot_fit'))}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        name_y = y + 7 if y <= 18 else y - 7

        fig.add_annotation(
            x=x,
            y=name_y,
            text=f"<b>{html.escape(name)}</b>",
            showarrow=False,
            font=dict(color="#cfcfcf", size=10),
            align="center",
            bgcolor="rgba(30,41,59,0.0)",
            borderpad=0,
        )

    fig.update_layout(
        height=760,
        xaxis=dict(range=[0, 100], visible=False, fixedrange=True),
        yaxis=dict(
            range=[0, 112],
            visible=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="#1e293b",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode="closest",
    )

    return fig
