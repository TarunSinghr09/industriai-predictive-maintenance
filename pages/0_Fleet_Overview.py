import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core
import ui

ui.setup("Fleet Overview", "🏭")
st.caption("Health of the whole plant at a glance. Demo fleet of 60 machines built from AI4I 2020 data.")

seed = st.sidebar.number_input("Fleet snapshot (seed)", 1, 999, 11, help="Change to load a different set of machines")
fleet = ui.get_fleet(int(seed))

counts = fleet.level.value_counts()
k = st.columns(5)
k[0].metric("Machines monitored", len(fleet))
k[1].metric("Critical", int(counts.get("CRITICAL", 0)))
k[2].metric("High risk", int(counts.get("HIGH", 0)))
k[3].metric("Avg failure risk", f"{fleet.prob.mean()*100:.1f}%")
k[4].metric("Expected failures", f"{fleet.prob.sum():.1f}")

a, b = st.columns([1, 1.6])
order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
with a:
    fig = go.Figure(go.Pie(labels=order, values=[int(counts.get(l, 0)) for l in order], hole=0.55,
                           marker_colors=[ui.COLORS[l] for l in order], sort=False))
    fig.update_layout(title="Fleet health", height=320, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")
with b:
    top = fleet.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=top.prob * 100, y=top.machine, orientation="h",
                           marker_color=[ui.COLORS[l] for l in top.level]))
    fig.update_layout(title="Top 10 machines by failure risk (%)", height=320, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")

st.subheader("All machines (sorted by risk)")
levels = st.multiselect("Filter by risk level", order, default=["CRITICAL", "HIGH", "MEDIUM"])
view = fleet[fleet.level.isin(levels)] if levels else fleet
st.dataframe(view.assign(prob=(view.prob * 100).round(1))[["machine", "type", "level", "prob", "reason", "urgency"]]
             .rename(columns={"prob": "risk %"}), width="stretch", hide_index=True)

st.subheader("Machine drill-down")
pick = st.selectbox("Select a machine", fleet.machine)
r = fleet[fleet.machine == pick].iloc[0]
row = pd.DataFrame([dict(air_temp=r.air_temp, proc_temp=r.proc_temp, speed=r.speed, torque=r.torque,
                         tool_wear=r.tool_wear, type_code=r.type_code)])
ui.show_result(*ui.assess(row))

