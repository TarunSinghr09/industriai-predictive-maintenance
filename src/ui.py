"""Shared Streamlit helpers used by every page."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core

COLORS = {"LOW": "#2e9e5b", "MEDIUM": "#e0b000", "HIGH": "#e8742a", "CRITICAL": "#d62f2f"}
TYPE_NAME = {0: "L", 1: "M", 2: "H"}


def setup(title: str, icon: str):
    st.set_page_config(page_title=f"{title} | INDUSTRIAI", page_icon=icon, layout="wide")
    st.sidebar.markdown("### 🏭 INDUSTRIAI")
    st.sidebar.caption("Industrial predictive maintenance platform")
    st.title(f"{icon} {title}")


@st.cache_resource
def get_predictor():
    return core.Predictor()


@st.cache_data
def get_stream():
    return core.load_data().sample(frac=1, random_state=7).reset_index(drop=True)


@st.cache_data
def get_fleet(seed: int = 11, n: int = 60) -> pd.DataFrame:
    """Demo fleet: n machines (a few failing units mixed in so the dashboard is interesting)."""
    pred = get_predictor()
    df = core.load_data()
    fail = df[df.failure == 1].sample(8, random_state=seed)
    ok = df[df.failure == 0].sample(n - 8, random_state=seed)
    f = pd.concat([fail, ok]).sample(frac=1, random_state=seed).reset_index(drop=True)
    probs = pred.predict(f)
    rows = []
    for i in range(len(f)):
        row = f.iloc[[i]]
        p = float(probs[i])
        expl = pred.explain(row, top=4)
        actions, urgency = core.recommend(f.iloc[i], p, expl)
        r = f.iloc[i]
        rows.append(dict(
            machine=r["Product ID"], type=TYPE_NAME[int(r.type_code)], prob=p,
            level=core.risk_level(p), status=core.status_of(p),
            reason=core.reason_text(expl), action=actions[0], urgency=urgency,
            air_temp=r.air_temp, proc_temp=r.proc_temp, speed=r.speed,
            torque=r.torque, tool_wear=r.tool_wear, type_code=int(r.type_code)))
    return pd.DataFrame(rows).sort_values("prob", ascending=False).reset_index(drop=True)


def shap_bar(expl_full):
    names = [e[0] for e in expl_full][::-1]
    vals = [e[2] for e in expl_full][::-1]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                           marker_color=["#d62f2f" if v > 0 else "#2e7bd6" for v in vals]))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0),
                      title="SHAP: why this prediction (red = raises risk)")
    return fig


def gauge(p):
    lvl = core.risk_level(p)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS[lvl]},
               "steps": [{"range": [0, 10], "color": "#e6f4ec"}, {"range": [10, 30], "color": "#fbf3d0"},
                         {"range": [30, 60], "color": "#fde3d0"}, {"range": [60, 100], "color": "#f9d4d4"}]},
        title={"text": "Failure probability"}))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=0))
    return fig


def assess(row_df: pd.DataFrame):
    pred = get_predictor()
    p = float(pred.predict(row_df)[0])
    expl_full = pred.explain(row_df, top=len(core.FEATURES))
    actions, urgency = core.recommend(row_df.iloc[0], p, expl_full)
    return p, expl_full, actions, urgency


def show_result(p, expl_full, actions, urgency):
    lvl = core.risk_level(p)
    color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "orange", "CRITICAL": "red"}[lvl]
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.plotly_chart(gauge(p), use_container_width=True)
        st.markdown(f"### Status: :{color}[{core.status_of(p)}]  \n**Risk level:** {lvl}")
    with c2:
        st.plotly_chart(shap_bar(expl_full), use_container_width=True)
    st.info("**Reason:** " + core.reason_text(expl_full))
    st.markdown(f"**Priority:** {urgency}")
    for a in actions:
        st.markdown(f"- 🔧 {a}")
