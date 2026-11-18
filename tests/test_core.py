import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

import core


def test_risk_levels():
    assert core.risk_level(0.02) == "LOW"
    assert core.risk_level(0.15) == "MEDIUM"
    assert core.risk_level(0.4) == "HIGH"
    assert core.risk_level(0.9) == "CRITICAL"


def test_rule_flags_overstrain_and_wear():
    row = pd.Series(dict(air_temp=300, proc_temp=310, speed=1500, torque=65, tool_wear=220, type_code=0))
    flags = core.rule_flags(row)
    assert "OSF" in flags and "TWF" in flags


def test_rule_flags_healthy():
    row = pd.Series(dict(air_temp=300, proc_temp=310, speed=1500, torque=40, tool_wear=50, type_code=0))
    assert core.rule_flags(row) == []


def test_dataset_loads():
    df = core.load_data()
    assert len(df) == 10000
    assert set(core.RAW.values()) <= set(df.columns)


@pytest.fixture(scope="module")
def predictor():
    return core.Predictor()


def test_prediction_range_and_ordering(predictor):
    healthy = pd.DataFrame([dict(air_temp=300, proc_temp=310, speed=1500, torque=40, tool_wear=50, type_code=0)])
    faulty = pd.DataFrame([dict(air_temp=300, proc_temp=310, speed=1400, torque=65, tool_wear=220, type_code=0)])
    ph, pf = predictor.predict(healthy)[0], predictor.predict(faulty)[0]
    assert 0 <= ph <= 1 and 0 <= pf <= 1
    assert pf > ph


def test_explanation_shape(predictor):
    row = pd.DataFrame([dict(air_temp=300, proc_temp=310, speed=1400, torque=65, tool_wear=220, type_code=0)])
    expl = predictor.explain(row, top=3)
    assert len(expl) == 3
    assert all(len(e) == 3 for e in expl)
