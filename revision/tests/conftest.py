from datetime import datetime, timedelta

import pytest

from revision.evaluation.contract import Contract, Outcome
from revision.evaluation.data import DataSpec, LoadScale, SplitPlan, Timeline, prepare
from revision.evaluation.selection import Candidate


@pytest.fixture
def timeline():
    start = datetime(2019, 4, 1)
    return Timeline("synthetic", tuple(start + timedelta(minutes=15 * i) for i in range(30)),
                    tuple(float(i + 1) for i in range(30)), timedelta(minutes=15), "fixture")


@pytest.fixture
def plan(timeline):
    ts = timeline.timestamps
    return SplitPlan(ts[0], ts[15], ts[23], ts[-1] + timeline.slot)


@pytest.fixture
def prepared(timeline, plan):
    scale = LoadScale.fit_training(plan.partition(timeline)["train"], 15)
    return prepare(timeline, plan, DataSpec(3, 4), scale=scale)


@pytest.fixture
def candidate():
    return Candidate("fixture-only", (("configuration", "a" * 64),))


@pytest.fixture
def contract():
    return Contract()


class RecordingController:
    """Test double only, not an implementation of any scientific controller."""

    def __init__(self, requests=(0,)):
        self.requests = requests
        self.observations = []
        self.resets = []

    def reset(self, *, seed):
        self.resets.append(seed)
        self.observations = []

    def predict(self, observation, *, deterministic):
        assert deterministic is True
        result = self.requests[len(self.observations) % len(self.requests)]
        self.observations.append(observation)
        return result


class SyntheticTwin:
    def outcome(self, action, load_mbps):
        return Outcome(4.0 + action + load_mbps / 100, 0.95 if action == 0 else 0.85)


@pytest.fixture
def controller():
    return RecordingController()


@pytest.fixture
def twin():
    return SyntheticTwin()
