import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

import core
import ui

ui.setup("Maintenance Planner", "🗓️")
st.caption("Turns risk into business decisions: which machines to service first, and is it worth it?")

fleet = ui.get_fleet(11)

with st.sidebar:
    st.subheader("Cost assumptions")
    cost_hr = st.number_input("Downtime cost per hour (INR)", 1000, 1000000, 50000, 1000)
    fail_hrs = st.number_input("Downtime if machine fails (hours)", 1.0, 48.0, 8.0, 0.5)
    plan_cost = st.number_input("Planned maintenance cost per machine (INR)", 100, 200000, 5000, 100)
    thr = st.slider("Service machines with risk above", 0.05, 0.9, 0.3, 0.05)

fleet = fleet.assign(
    exp_loss=core.expected_loss(fleet.prob, cost_hr, fail_hrs),
    priority=lambda d: d.exp_loss.rank(ascending=False, method="first").astype(int))
sel = fleet[fleet.prob >= thr]
avoided = sel.exp_loss.sum()
spend = len(sel) * plan_cost

k = st.columns(4)
k[0].metric("Machines to service", len(sel))
k[1].metric("Maintenance spend", f"INR {spend:,.0f}")
k[2].metric("Expected loss avoided", f"INR {avoided:,.0f}")
k[3].metric("Net benefit", f"INR {avoided - spend:,.0f}")
st.caption("Expected loss = failure probability x downtime hours x cost per hour. Assumes servicing prevents the failure.")

st.subheader("Work orders (priority order)")
orders = sel.sort_values("exp_loss", ascending=False).assign(
    status="Open", risk_pct=lambda d: (d.prob * 100).round(1), exp_loss=lambda d: d.exp_loss.round(0))
edited = st.data_editor(
    orders[["priority", "machine", "level", "risk_pct", "exp_loss", "urgency", "action", "status"]],
    column_config={"status": st.column_config.SelectboxColumn(
        "status", options=["Open", "Scheduled", "In progress", "Done"], required=True)},
    disabled=["priority", "machine", "level", "risk_pct", "exp_loss", "urgency", "action"],
    hide_index=True, width="stretch")
st.download_button("Download work orders (CSV)", edited.to_csv(index=False).encode(),
                   "work_orders.csv", "text/csv")
done = int((edited.status == "Done").sum())
st.progress(done / max(len(edited), 1), text=f"{done} of {len(edited)} work orders completed")
