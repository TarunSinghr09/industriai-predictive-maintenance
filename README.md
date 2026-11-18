# INDUSTRIAI - Industrial Predictive Maintenance (prototype)

Predicts machine failure, explains it with SHAP, prioritizes risk and recommends maintenance.
Dataset: AI4I 2020 (UCI) in `data/ai4i2020.csv`.

## Run
```
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python src\core.py            # trains RF + XGBoost, saves best to models/
.venv\Scripts\python -m streamlit run app.py
```

## Structure
- `src/core.py` - data, feature engineering, models, SHAP, risk levels, recommendations
- `app.py` - Streamlit UI: Live Monitoring (dataset replayed row by row), Manual Check form, Model Info
