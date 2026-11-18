import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import streamlit as st

import core
import ui

ui.setup("Machine Check", "🧪")
st.caption("Enter sensor values and get instant status, risk, SHAP reason and maintenance advice.")

PRESETS = {  # air K, process K, rpm, torque Nm, wear min, type
    "Custom / healthy": (300.0, 310.0, 1500, 40.0, 50, "L"),
    "Overstrain failure": (300.0, 310.5, 1400, 65.0, 220, "L"),
    "Heat dissipation failure": (302.0, 309.0, 1300, 40.0, 100, "M"),
    "Tool wear failure": (299.0, 309.0, 1500, 40.0, 230, "H"),
    "Power failure": (300.0, 310.0, 2800, 60.0, 80, "L"),
}
preset = st.selectbox("Quick scenario", list(PRESETS))
a0, p0, s0, t0, w0, ty0 = PRESETS[preset]

with st.form("manual"):
    a, b, c3 = st.columns(3)
    air = a.number_input("Air temperature (K)", 290.0, 310.0, a0, 0.1, key=f"air{preset}")
    proc = b.number_input("Process temperature (K)", 300.0, 320.0, p0, 0.1, key=f"proc{preset}")
    typ = c3.selectbox("Product type", ["L", "M", "H"], index=["L", "M", "H"].index(ty0), key=f"typ{preset}")
    speed = a.number_input("Rotational speed (rpm)", 1000, 3000, int(s0), 10, key=f"spd{preset}")
    torque = b.number_input("Torque (Nm)", 3.0, 80.0, t0, 0.5, key=f"tq{preset}")
    wear = c3.number_input("Tool wear (min)", 0, 300, int(w0), 1, key=f"wr{preset}")
    ok = st.form_submit_button("Analyse", type="primary")

if ok:
    row = pd.DataFrame([dict(air_temp=air, proc_temp=proc, speed=speed, torque=torque,
                             tool_wear=wear, type_code=core.TYPE_MAP[typ])])
    ui.show_result(*ui.assess(row))
