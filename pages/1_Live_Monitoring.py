import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core
import ui

ui.setup("Live Monitoring", "📡")
st.caption("No physical sensors, so dataset rows are streamed one by one as simulated real-time readings.")

pred = ui.get_predictor()
stream = ui.get_stream()

c = st.columns(4)
delay = c[0].slider("Speed (sec/reading)", 0.05, 1.0, 0.2, 0.05)
window = c[1].slider("Graph window (readings)", 30, 200, 80, 10)
if c[2].button("▶ Start / Resume", type="primary"):
    st.session_state.running = True
if c[3].button("⏹ Stop"):
    st.session_state.running = False
st.session_state.setdefault("running", False)
st.session_state.setdefault("idx", 0)
st.session_state.setdefault("log", [])
if st.button("↺ Reset"):
    st.session_state.update(idx=0, log=[], running=False)

kpi, charts, alerts = st.container(), st.container(), st.container()


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
        f = go.Figure(go.Scatter(x=w.t, y=w.prob * 100, mode="lines+markers", line=dict(color="#d62f2f")))
        f.add_hline(y=30, line_dash="dot", annotation_text="HIGH")
        f.update_layout(title="Failure risk (%)", height=280, margin=dict(l=0, r=0, t=40, b=0), yaxis_range=[0, 100])
        a.plotly_chart(f, width="stretch")
        g = go.Figure()
        g.add_trace(go.Scatter(x=w.t, y=w.torque, name="Torque (Nm)"))
        g.add_trace(go.Scatter(x=w.t, y=w.proc_temp - 273.15, name="Process temp (C)", yaxis="y2"))
        g.update_layout(title="Torque & temperature", height=280, margin=dict(l=0, r=0, t=40, b=0),
                        yaxis2=dict(overlaying="y", side="right"), legend=dict(orientation="h"))
        b.plotly_chart(g, width="stretch")
        a2, b2 = st.columns(2)
        h = go.Figure(go.Scatter(x=w.t, y=w.speed, line=dict(color="#2e7bd6")))
        h.update_layout(title="Rotational speed (rpm)", height=240, margin=dict(l=0, r=0, t=40, b=0))
        a2.plotly_chart(h, width="stretch")
        i = go.Figure(go.Scatter(x=w.t, y=w.tool_wear, line=dict(color="#8a5cd6")))
        i.update_layout(title="Tool wear (min)", height=240, margin=dict(l=0, r=0, t=40, b=0))
        b2.plotly_chart(i, width="stretch")
    with alerts:
        st.subheader("Risk-prioritized maintenance queue")
        q = log[log.level.isin(["CRITICAL", "HIGH"])].sort_values("prob", ascending=False).head(8)
        if q.empty:
            st.success("No high-risk machines so far.")
        else:
            q = q.assign(prob=(q.prob * 100).round(1))
            st.dataframe(q[["machine", "level", "prob", "reason", "action"]].rename(columns={"prob": "risk %"}),
                         width="stretch", hide_index=True)


if st.session_state.running:
    i = st.session_state.idx % len(stream)
    row = stream.iloc[[i]]
    p, expl, actions, urgency = ui.assess(row)
    r = row.iloc[0]
    st.session_state.log.append(dict(
        t=st.session_state.idx, machine=r["Product ID"], prob=p, level=core.risk_level(p),
        torque=r.torque, speed=r.speed, tool_wear=r.tool_wear, proc_temp=r.proc_temp,
        reason=core.reason_text(expl), action=actions[0], actual=int(r.failure)))
    st.session_state.idx += 1
    st.session_state.log = st.session_state.log[-1000:]
    render()
    time.sleep(delay)
    st.rerun()
else:
    render()
