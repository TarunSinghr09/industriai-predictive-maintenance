"""Core logic: data, model, SHAP explanation, risk scoring, recommendations."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ai4i2020.csv"
MODEL = ROOT / "models" / "model.joblib"

RAW = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "proc_temp",
    "Rotational speed [rpm]": "speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}
FEATURES = ["air_temp", "proc_temp", "speed", "torque", "tool_wear",
            "temp_diff", "power_w", "wear_torque", "type_code"]
LABELS = {
    "air_temp": "Air temperature", "proc_temp": "Process temperature",
    "speed": "Rotational speed", "torque": "Torque", "tool_wear": "Tool wear",
    "temp_diff": "Temp difference (proc - air)", "power_w": "Mechanical power",
    "wear_torque": "Wear x Torque (overstrain)", "type_code": "Product type",
}
FAILURE_MODES = {
    "TWF": "Tool Wear Failure", "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure", "OSF": "Overstrain Failure", "RNF": "Random Failure",
}
TYPE_MAP = {"L": 0, "M": 1, "H": 2}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA).rename(columns=RAW)
    df["Type"] = df["Type"].map(TYPE_MAP)
    return df.rename(columns={"Type": "type_code", "Machine failure": "failure"})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["temp_diff"] = d["proc_temp"] - d["air_temp"]
    d["power_w"] = d["torque"] * d["speed"] * 2 * np.pi / 60
    d["wear_torque"] = d["tool_wear"] * d["torque"]
    return d


def train(save: bool = True) -> dict:
    df = add_features(load_data())
    X, y = df[FEATURES], df["failure"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pos_w = (ytr == 0).sum() / (ytr == 1).sum()
    candidates = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
            scale_pos_weight=pos_w, eval_metric="logloss", random_state=42),
    }
    scores, fitted = {}, {}
    for name, m in candidates.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        scores[name] = dict(auc=roc_auc_score(yte, p), recall=recall_score(yte, pred),
                            precision=precision_score(yte, pred, zero_division=0))
        fitted[name] = m
    best = max(scores, key=lambda k: scores[k]["auc"])
    bundle = {"model": fitted[best], "name": best, "scores": scores}
    if save:
        MODEL.parent.mkdir(exist_ok=True)
        joblib.dump(bundle, MODEL)
    return bundle


def load_bundle() -> dict:
    if not MODEL.exists():
        return train()
    return joblib.load(MODEL)


class Predictor:
    def __init__(self):
        self.bundle = load_bundle()
        self.model = self.bundle["model"]
        self.explainer = shap.TreeExplainer(self.model)

    def prepare(self, rows: pd.DataFrame) -> pd.DataFrame:
        return add_features(rows)[FEATURES]

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.prepare(rows))[:, 1]

    def shap_values(self, rows: pd.DataFrame) -> np.ndarray:
        sv = self.explainer.shap_values(self.prepare(rows))
        if isinstance(sv, list):          # older shap: list per class
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                  # newer shap: (n, features, classes)
            sv = sv[:, :, 1]
        return sv

    def explain(self, row: pd.DataFrame, top: int = 4) -> list[tuple[str, float, float]]:
        """Return [(feature label, value, shap contribution)] sorted by |impact|."""
        sv = self.shap_values(row)[0]
        X = self.prepare(row).iloc[0]
        order = np.argsort(-np.abs(sv))[:top]
        return [(LABELS[FEATURES[i]], float(X.iloc[i]), float(sv[i])) for i in order]


def risk_level(p: float) -> str:
    if p >= 0.6:
        return "CRITICAL"
    if p >= 0.3:
        return "HIGH"
    if p >= 0.1:
        return "MEDIUM"
    return "LOW"


def status_of(p: float) -> str:
    return {"CRITICAL": "Failure imminent", "HIGH": "Warning",
            "MEDIUM": "Watch", "LOW": "Healthy"}[risk_level(p)]


def rule_flags(r: pd.Series) -> list[str]:
    """Physics-based failure-mode checks (from the AI4I dataset definition)."""
    flags = []
    power = r["torque"] * r["speed"] * 2 * np.pi / 60
    if r["tool_wear"] >= 200:
        flags.append("TWF")
    if (r["proc_temp"] - r["air_temp"]) < 8.6 and r["speed"] < 1380:
        flags.append("HDF")
    if power < 3500 or power > 9000:
        flags.append("PWF")
    limit = {0: 11000, 1: 12000, 2: 13000}[int(r["type_code"])]
    if r["tool_wear"] * r["torque"] > limit:
        flags.append("OSF")
    return flags


RECOMMENDATIONS = {
    "TWF": "Replace the cutting tool at the next stop - wear is near the 200-240 min failure window.",
    "HDF": "Check coolant flow and cleaning of heat exchanger; raise rotational speed or reduce load to restore the temperature gap.",
    "PWF": "Inspect drive/motor - power is outside the safe 3.5-9 kW band. Reduce torque or speed set-point.",
    "OSF": "Reduce torque and tool load; wear x torque is above the overstrain limit for this product type.",
}


def recommend(row: pd.Series, p: float, expl: list) -> tuple[list[str], str]:
    flags = rule_flags(row)
    actions = [RECOMMENDATIONS[f] for f in flags]
    if not actions and p >= 0.1:
        top = expl[0][0]
        actions.append(f"Model flags unusual pattern, mainly driven by {top.lower()} - schedule inspection.")
    lvl = risk_level(p)
    if lvl == "CRITICAL":
        urgency = "Stop machine / maintain immediately (< 1 hour)"
    elif lvl == "HIGH":
        urgency = "Schedule maintenance within 24 hours"
    elif lvl == "MEDIUM":
        urgency = "Plan inspection in next maintenance window"
    else:
        urgency = "No action - continue monitoring"
        actions = actions or ["All parameters within normal range."]
    return actions, urgency


def reason_text(expl: list) -> str:
    parts = []
    for name, val, s in expl[:3]:
        arrow = "raises" if s > 0 else "lowers"
        parts.append(f"{name} ({val:.1f}) {arrow} failure risk")
    return "; ".join(parts)


if __name__ == "__main__":
    b = train()
    print("Best model:", b["name"])
    for k, v in b["scores"].items():
        print(f"{k:14s} AUC={v['auc']:.3f} recall={v['recall']:.3f} precision={v['precision']:.3f}")
