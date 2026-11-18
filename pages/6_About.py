import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

import ui

ui.setup("About", "ℹ️")
st.markdown("""
**INDUSTRIAI** predicts machine failure from sensor readings, explains each prediction with SHAP,
prioritizes machines by risk and business impact, and recommends maintenance actions.

### Pipeline
`Sensor readings -> feature engineering -> XGBoost / Random Forest -> failure probability -> SHAP reasons -> risk level -> maintenance recommendation -> cost-based work orders`

### Pages
- **Fleet Overview:** plant health, top-risk machines, drill-down per machine
- **Live Monitoring:** dataset replayed row by row as a simulated real-time stream
- **Machine Check:** enter values manually, get instant status and reasons
- **Maintenance Planner:** expected loss, work orders, net benefit of servicing
- **Model Performance:** confusion matrix, ROC, PR, threshold tuning, SHAP
- **Data Explorer:** failure modes, distributions, raw data

### Limitations (honest)
- The AI4I 2020 dataset is synthetic; real plants will need retraining on real sensor data.
- Live mode is a simulation, not connected to real sensors.
- The demo fleet is a sample from the same dataset the model was trained on, so it illustrates the workflow, it is not a performance claim. See **Model Performance** for hold-out results.
- Cost figures in the planner are adjustable assumptions.

### Stack
Python, Pandas, scikit-learn, XGBoost, SHAP, Streamlit, Plotly, pytest, GitHub Actions, Docker.
""")
