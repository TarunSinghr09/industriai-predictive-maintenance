import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core
import ui

ui.setup("Data Explorer", "🔎")
df = core.load_data()
st.caption("AI4I 2020 Predictive Maintenance dataset (UCI): 10,000 synthetic machine records.")

k = st.columns(4)
k[0].metric("Rows", f"{len(df):,}")
k[1].metric("Failures", int(df.failure.sum()))
k[2].metric("Failure rate", f"{df.failure.mean()*100:.2f}%")
k[3].metric("Features", len(core.RAW))

t1, t2, t3 = st.tabs(["Failure modes", "Distributions", "Raw data"])
with t1:
    modes = {core.FAILURE_MODES[m]: int(df[m].sum()) for m in core.FAILURE_MODES}
    fig = go.Figure(go.Bar(x=list(modes), y=list(modes.values())))
    fig.update_layout(title="Failures by mode", height=340, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Class imbalance (about 96.6% healthy) is why the model uses class weighting and is judged on recall/precision.")
with t2:
    feat = st.selectbox("Feature", list(core.RAW.values()))
    d = df.assign(outcome=df.failure.map({0: "Healthy", 1: "Failed"}))
    fig = px.histogram(d, x=feat, color="outcome", barmode="overlay", opacity=0.65, nbins=50,
                       color_discrete_map={"Healthy": "#2e9e5b", "Failed": "#d62f2f"})
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, width="stretch")
    corr = df[list(core.RAW.values()) + ["failure"]].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(title="Correlation", height=420, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")
with t3:
    only = st.checkbox("Show failures only")
    st.dataframe(df[df.failure == 1] if only else df, width="stretch", hide_index=True)
