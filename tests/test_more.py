import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

import core
import ui


def test_expected_loss_formula():
    assert core.expected_loss(0.5, 50000, 8) == 200000
    assert core.expected_loss(0.0, 50000, 8) == 0


def test_status_matches_level():
    assert core.status_of(0.01) == "Healthy"
    assert core.status_of(0.95) == "Failure imminent"


def test_holdout_split_is_stratified():
    X, y = core.holdout()
    assert len(X) == 2500
    assert int(y.sum()) == 85
    assert list(X.columns) == core.FEATURES


def test_fleet_structure():
    fleet = ui.get_fleet(11)
    assert len(fleet) == 60
    assert fleet.prob.is_monotonic_decreasing
    assert set(fleet.level) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert {"machine", "prob", "reason", "action", "urgency"} <= set(fleet.columns)


@pytest.fixture(scope="module")
def predictor():
    return core.Predictor()


def test_recommendation_for_overstrain(predictor):
    row = pd.DataFrame([dict(air_temp=300, proc_temp=310.5, speed=1400, torque=65, tool_wear=220, type_code=0)])
    p = float(predictor.predict(row)[0])
    expl = predictor.explain(row)
    actions, urgency = core.recommend(row.iloc[0], p, expl)
    assert any("overstrain" in a.lower() for a in actions)
    assert "immediately" in urgency.lower()


def test_healthy_machine_gets_no_action(predictor):
    row = pd.DataFrame([dict(air_temp=300, proc_temp=310, speed=1500, torque=40, tool_wear=50, type_code=0)])
    p = float(predictor.predict(row)[0])
    actions, urgency = core.recommend(row.iloc[0], p, predictor.explain(row))
    assert "no action" in urgency.lower()
    assert actions == ["All parameters within normal range."]
