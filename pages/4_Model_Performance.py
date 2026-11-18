import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

import core
import ui

ui.setup("Model Performance", "📊")
pred = ui.get_predictor()
st.caption(f"Deployed model: **{pred.bundle['name']}** · evaluated on the 25% hold-out set the model never saw.")

Xte, yte = core.holdout()
p = pred.model.predict_proba(Xte)[:, 1]

thr = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05,
                help="Lower = catch more failures but more false alarms")
yp = (p >= thr).astype(int)
k = st.columns(5)
k[0].metric("ROC-AUC", f"{roc_auc_score(yte, p):.3f}")
k[1].metric("Avg precision", f"{average_precision_score(yte, p):.3f}")
k[2].metric("Recall", f"{recall_score(yte, yp):.3f}")
k[3].metric("Precision", f"{precision_score(yte, yp, zero_division=0):.3f}")
k[4].metric("F1", f"{f1_score(yte, yp):.3f}")

a, b = st.columns(2)
cm = confusion_matrix(yte, yp)
with a:
    fig = go.Figure(go.Heatmap(z=cm, x=["Pred OK", "Pred Fail"], y=["Actual OK", "Actual Fail"],
                               text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title="Confusion matrix", height=330, margin=dict(l=0, r=0, t=40, b=0), yaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
with b:
    fpr, tpr, _ = roc_curve(yte, p)
    fig = go.Figure(go.Scatter(x=fpr, y=tpr, name="ROC"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot"), name="random"))
    fig.update_layout(title="ROC curve", height=330, xaxis_title="False positive rate",
                      yaxis_title="True positive rate", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

c, d = st.columns(2)
with c:
    pr, rc, _ = precision_recall_curve(yte, p)
    fig = go.Figure(go.Scatter(x=rc, y=pr))
    fig.update_layout(title="Precision-recall curve", height=330, xaxis_title="Recall", yaxis_title="Precision",
                      margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
with d:
    st.markdown("**Model comparison**")
    st.dataframe(pd.DataFrame(pred.bundle["scores"]).T.round(3), use_container_width=True)
    st.caption("Failures are rare (~3.4%), so recall and precision matter more than accuracy.")

st.subheader("Explainability (SHAP)")
sample = Xte.sample(500, random_state=1)
sv = pred.explainer.shap_values(sample)
sv = sv[1] if isinstance(sv, list) else np.asarray(sv)
if sv.ndim == 3:
    sv = sv[:, :, 1]
imp = np.abs(sv).mean(0)
order = np.argsort(imp)
e, f = st.columns(2)
with e:
    fig = go.Figure(go.Bar(x=imp[order], y=[core.LABELS[core.FEATURES[i]] for i in order], orientation="h"))
    fig.update_layout(title="Global importance (mean |SHAP|)", height=380, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
with f:
    feat = st.selectbox("Dependence plot feature", core.FEATURES, format_func=core.LABELS.get)
    j = core.FEATURES.index(feat)
    fig = go.Figure(go.Scatter(x=sample[feat], y=sv[:, j], mode="markers", marker=dict(size=5, opacity=0.6)))
    fig.update_layout(title=f"How {core.LABELS[feat]} moves risk", xaxis_title=core.LABELS[feat],
                      yaxis_title="SHAP value", height=330, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
