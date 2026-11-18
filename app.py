import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="INDUSTRIAI", page_icon="🏭", layout="wide")

COLORS = {"LOW": "#2e9e5b", "MEDIUM": "#e0b000", "HIGH": "#e8742a", "CRITICAL": "#d62f2f"}


@st.cache_resource
def get_predictor():
    return core.Predictor()


@st.cache_data
def get_stream():
    df = core.load_data().sample(frac=1, random_state=7).reset_index(drop=True)
    return df


pred = get_predictor()
stream = get_stream()


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
    p = float(pred.predict(row_df)[0])
    expl_full = pred.explain(row_df, top=len(core.FEATURES))
    actions, urgency = core.recommend(row_df.iloc[0], p, expl_full)
    return p, expl_full, actions, urgency


def show_result(p, expl_full, actions, urgency):
    lvl = core.risk_level(p)
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.plotly_chart(gauge(p), use_container_width=True)
        st.markdown(f"### Status: :{ {'LOW':'green','MEDIUM':'orange','HIGH':'orange','CRITICAL':'red'}[lvl] }[{core.status_of(p)}]  \n**Risk level:** {lvl}")
    with c2:
        st.plotly_chart(shap_bar(expl_full), use_container_width=True)
    st.info("**Reason:** " + core.reason_text(expl_full))
    st.markdown(f"**Priority:** {urgency}")
    for a in actions:
        st.markdown(f"- 🔧 {a}")


st.title("🏭 INDUSTRIAI - Predictive Maintenance Platform")
st.caption(f"AI4I 2020 dataset · model: {pred.bundle['name']} · SHAP explainability · simulated live sensors")

tab1, tab2, tab3 = st.tabs(["📡 Live Monitoring", "🧪 Manual Check", "📊 Model Info"])

# ---------------------------------------------------------------- live
with tab1:
    c = st.columns(4)
    speed_delay = c[0].slider("Speed (sec/reading)", 0.05, 1.0, 0.2, 0.05)
    window = c[1].slider("Graph window (readings)", 30, 200, 80, 10)
    start = c[2].button("▶ Start / Resume", type="primary")
    stop = c[3].button("⏹ Stop")

    st.session_state.setdefault("running", False)
    st.session_state.setdefault("idx", 0)
    st.session_state.setdefault("log", [])
    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False
    if st.button("↺ Reset"):
        st.session_state.update(idx=0, log=[], running=False)

    kpi = st.container()
    charts = st.container()
    alerts = st.container()

    def render():
        log = pd.DataFrame(st.session_state.log)
        if log.empty:
            kpi.info("Press **Start** to begin streaming dataset rows as machine sensor readings.")
            return
        last = log.iloc[-1]
        with kpi:
            k = st.columns(5)
            k[0].metric("Readings", len(log))
            k[1].metric("Current risk", f"{last.prob*100:.1f}%", last.level)
            k[2].metric("Torque (Nm)", f"{last.torque:.1f}")
            k[3].metric("Speed (rpm)", f"{last.speed:.0f}")
            k[4].metric("Tool wear (min)", f"{last.tool_wear:.0f}")
        w = log.tail(window)
        with charts:
            a, b = st.columns(2)
            f = go.Figure(go.Scatter(x=w.t, y=w.prob * 100, mode="lines+markers",
                                     line=dict(color="#d62f2f"), name="risk"))
            f.add_hline(y=30, line_dash="dot", annotation_text="HIGH")
            f.update_layout(title="Failure risk (%)", height=280, margin=dict(l=0, r=0, t=40, b=0), yaxis_range=[0, 100])
            a.plotly_chart(f, use_container_width=True)
            g = go.Figure()
            g.add_trace(go.Scatter(x=w.t, y=w.torque, name="Torque (Nm)"))
            g.add_trace(go.Scatter(x=w.t, y=w.proc_temp - 273.15, name="Process temp (°C)", yaxis="y2"))
            g.update_layout(title="Torque & temperature", height=280, margin=dict(l=0, r=0, t=40, b=0),
                            yaxis2=dict(overlaying="y", side="right"), legend=dict(orientation="h"))
            b.plotly_chart(g, use_container_width=True)
            a2, b2 = st.columns(2)
            h = go.Figure(go.Scatter(x=w.t, y=w.speed, line=dict(color="#2e7bd6")))
            h.update_layout(title="Rotational speed (rpm)", height=240, margin=dict(l=0, r=0, t=40, b=0))
            a2.plotly_chart(h, use_container_width=True)
            i = go.Figure(go.Scatter(x=w.t, y=w.tool_wear, line=dict(color="#8a5cd6")))
            i.update_layout(title="Tool wear (min)", height=240, margin=dict(l=0, r=0, t=40, b=0))
            b2.plotly_chart(i, use_container_width=True)
        with alerts:
            st.subheader("Risk-prioritized maintenance queue")
            q = log[log.level.isin(["CRITICAL", "HIGH"])].sort_values("prob", ascending=False).head(8)
            if q.empty:
                st.success("No high-risk machines so far.")
            else:
                q = q.assign(prob=(q.prob * 100).round(1))
                st.dataframe(q[["machine", "level", "prob", "reason", "action"]]
                             .rename(columns={"prob": "risk %"}), use_container_width=True, hide_index=True)

    if st.session_state.running:
        # process a small batch per rerun, redraw, then rerun -> live feel
        for _ in range(1):
            i = st.session_state.idx % len(stream)
            row = stream.iloc[[i]]
            p, expl, actions, urgency = assess(row)
            r = row.iloc[0]
            st.session_state.log.append(dict(
                t=st.session_state.idx, machine=r["Product ID"], prob=p, level=core.risk_level(p),
                torque=r.torque, speed=r.speed, tool_wear=r.tool_wear, proc_temp=r.proc_temp,
                reason=core.reason_text(expl), action=actions[0], actual=int(r.failure)))
            st.session_state.idx += 1
        st.session_state.log = st.session_state.log[-1000:]
        render()
        time.sleep(speed_delay)
        st.rerun()
    else:
        render()

# ---------------------------------------------------------------- form
with tab2:
    st.subheader("Enter machine readings")
    with st.form("manual"):
        a, b, c3 = st.columns(3)
        air = a.number_input("Air temperature (K)", 290.0, 310.0, 300.0, 0.1)
        proc = b.number_input("Process temperature (K)", 300.0, 320.0, 310.0, 0.1)
        typ = c3.selectbox("Product type", ["L", "M", "H"])
        speed = a.number_input("Rotational speed (rpm)", 1000, 3000, 1500, 10)
        torque = b.number_input("Torque (Nm)", 3.0, 80.0, 40.0, 0.5)
        wear = c3.number_input("Tool wear (min)", 0, 300, 100, 1)
        ok = st.form_submit_button("Analyse", type="primary")
    if ok:
        row = pd.DataFrame([dict(air_temp=air, proc_temp=proc, speed=speed, torque=torque,
                                 tool_wear=wear, type_code=core.TYPE_MAP[typ])])
        show_result(*assess(row))
    st.caption("Try: torque 65, wear 220 (overstrain) or speed 1300 with temp gap 7 K (heat failure).")

# ---------------------------------------------------------------- info
with tab3:
    st.subheader("Model comparison (25% hold-out)")
    sc = pd.DataFrame(pred.bundle["scores"]).T.round(3)
    st.dataframe(sc, use_container_width=True)
    st.write(f"Deployed model: **{pred.bundle['name']}** (best ROC-AUC).")
    imp = np.abs(pred.shap_values(core.add_features(core.load_data()).sample(500, random_state=1))).mean(0)
    order = np.argsort(imp)
    fig = go.Figure(go.Bar(x=imp[order], y=[core.LABELS[core.FEATURES[i]] for i in order], orientation="h"))
    fig.update_layout(title="Global feature importance (mean |SHAP|)", height=380, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
