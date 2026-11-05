from dataclasses import replace

import pytest

from revision.evaluation.contract import (
    Contract, Forecast, Outcome, PersistenceForecast, State, account, actuate,
    causal_forecast, evaluate, require_comparable,
)
from revision.evaluation.data import FitProvenance
from revision.evaluation.selection import ValidationScore, freeze_selection
from revision.tests.conftest import RecordingController


def test_one_step_forecast_targets_upcoming_interval_using_completed_history(prepared):
    h = prepared.horizons["validation"]
    context = h.context(h.partition.start)
    f = causal_forecast(context, PersistenceForecast(), None)
    assert f.issued_at == context.decision_at == f.target_at
    assert f.observed_through == context.completed_interval_at
    assert f.load_mbps == context.forecast_input_mbps[-1]
    assert f.load_mbps != h.partition.timeline.load_mbps[h.partition.start]


@pytest.mark.parametrize("field", ["issued_at", "observed_through", "target_at"])
def test_shifted_forecast_is_rejected(prepared, field):
    context = prepared.horizons["validation"].context(15)

    class Misaligned(PersistenceForecast):
        def predict(self, c):
            return replace(super().predict(c), **{field: c.decision_at + prepared.partitions["train"].timeline.slot})

    with pytest.raises(ValueError, match="misaligned"):
        causal_forecast(context, Misaligned(), None)


def test_learned_forecaster_requires_matching_training_provenance(prepared):
    context = prepared.horizons["validation"].context(15)

    class LearnedFixture(PersistenceForecast):
        fit = FitProvenance.from_training(prepared.partitions["train"])

    model = LearnedFixture()
    with pytest.raises(ValueError, match="provenance"):
        causal_forecast(context, model, None)
    assert causal_forecast(context, model, model.fit).target_at == context.decision_at
    with pytest.raises(ValueError, match="not available"):
        causal_forecast(prepared.horizons["train"].context(4), model, model.fit)


def test_future_target_perturbation_leaves_decision_inputs_and_forecast_unchanged(prepared):
    h = prepared.horizons["validation"]
    altered = replace(h.partition.timeline, load_mbps=h.partition.timeline.load_mbps[:15] + (1e8,) * 15)
    alternative = replace(h, partition=replace(h.partition, timeline=altered))
    assert h.context(15) == alternative.context(15)
    assert PersistenceForecast().predict(h.context(15)) == PersistenceForecast().predict(alternative.context(15))


def test_cooldown_observed_counter_matches_guard_and_change_at_t_plus_four(contract):
    first = actuate(State(0, 4), 1, contract)
    assert first.executed == 1 and first.next_state.elapsed_slots == 1
    state = first.next_state
    for elapsed in (1, 2, 3):
        assert state.elapsed_slots == elapsed
        transition = actuate(state, 0, contract)
        assert transition.blocked and transition.executed == 1
        state = transition.next_state
    assert state.elapsed_slots == 4
    assert actuate(state, 0, contract).executed == 0


def test_blocked_request_is_discarded_and_unchanged_action_does_not_reset(contract):
    state = actuate(State(0, 4), 1, contract).next_state
    blocked = actuate(state, 2, contract)
    assert blocked.blocked and not blocked.type_switch and not blocked.scale_up
    state = blocked.next_state
    for _ in range(5):
        transition = actuate(state, 1, contract)
        assert transition.executed == 1  # Discarded USR(2) is never queued.
        state = transition.next_state
    assert state.elapsed_slots == 4


def test_zero_cooldown_allows_consecutive_changes(contract):
    c = replace(contract, cooldown_slots=0)
    first = actuate(State(0, 0), 1, c)
    assert actuate(first.next_state, 2, c).executed == 2


@pytest.mark.parametrize("old,new,cost,launched,type_switch,up,down", [
    (0, 1, .012, 1, True, False, False),
    (0, 2, .024, 2, True, False, False),
    (1, 0, .030, 0, True, False, False),
    (2, 0, .030, 0, True, False, False),
    (1, 2, .012, 1, False, True, False),
    (2, 1, .003, 0, False, False, True),
])
def test_submitted_launch_cost_and_event_definitions(contract, old, new, cost, launched, type_switch, up, down):
    tr = actuate(State(old, 4), new, contract)
    assert (tr.usr_instances_launched, tr.type_switch, tr.scale_up, tr.scale_down) == (launched, type_switch, up, down)
    assert account(Outcome(4, .95), 10, 900, tr, contract).orchestration_term == pytest.approx(cost)


def test_energy_units_qos_threshold_and_blocked_action_accounting(contract):
    tr = actuate(State(0, 0), 2, contract)
    metrics = account(Outcome(8, .90), 4, 900, tr, contract)
    assert metrics.energy_wh == 2
    assert metrics.offered_mbit == 3600
    assert metrics.sec_j_per_mbit == 2
    assert not metrics.qos_violation  # Equality satisfies the threshold.
    assert metrics.orchestration_term == 0
    assert metrics.reward == -200
    assert account(Outcome(8, .89), 4, 1800, tr, contract).energy_wh == 4
    zero = account(Outcome(1, .8), 0, 900, tr, contract)
    assert zero.sec_j_per_mbit == 1e6
    assert zero.qos_shortfall == pytest.approx(.1)


@pytest.mark.parametrize("power,qos", [(-1, .9), (float("nan"), .9), (1, 1.01), (1, float("nan"))])
def test_invalid_twin_predictions_fail_without_silent_clipping(power, qos):
    with pytest.raises(ValueError):
        Outcome(power, qos)


@pytest.mark.parametrize("action", [-1, 3, True, 1.0])
def test_invalid_action_rejected(contract, action):
    with pytest.raises(ValueError):
        actuate(State(0, 4), action, contract)


@pytest.mark.parametrize("schema", ["instant", "history", "forecast", "hybrid"])
def test_schema_information_and_previous_outcomes_are_causal(prepared, candidate, contract, twin, schema):
    ctl = RecordingController()
    result = evaluate(prepared.horizons["validation"], ctl, twin, candidate=candidate,
                      contract=contract, schema=schema, forecaster=PersistenceForecast())
    first, second = ctl.observations[:2]
    assert first.previous_outcome is None
    assert second.previous_outcome == result.steps[0].outcome
    assert first.current_load_mbps == prepared.partitions["validation"].timeline.load_mbps[14]
    assert first.available_actions == (0, 1, 2)
    assert bool(first.history_mbps) == (schema in ("history", "hybrid"))
    assert (first.forecast is not None) == (schema in ("forecast", "hybrid"))
    assert not hasattr(first, "context")  # No schema bypass via the full history context.
    assert not hasattr(first, "load_mbps")  # No upcoming actual load in the observation.


@pytest.mark.parametrize("adapter_name", ["ppo", "recurrent_ppo", "rule", "static"])
def test_all_controller_interfaces_use_identical_final_timestamps(prepared, candidate, contract, twin, adapter_name):
    # Names denote future adapters. The double tests the shared runner, not policy behavior.
    ctl = RecordingController()
    validation = evaluate(prepared.horizons["validation"], ctl, twin, candidate=candidate,
                          contract=contract, schema="instant")
    score = ValidationScore.from_result(candidate, validation, "total_reward")
    frozen = freeze_selection((score,), validation=prepared.horizons["validation"],
                              final_test=prepared.horizons["final_test"], objective="total_reward")
    result = evaluate(prepared.horizons["final_test"], ctl, twin, candidate=candidate,
                      contract=contract, schema="instant", frozen_selection=frozen)
    assert tuple(o.decision_at for o in ctl.observations) == prepared.horizons["final_test"].timestamps
    assert tuple(s.load_mbps for s in result.steps) == prepared.partitions["final_test"].loads


def test_deterministic_replay_and_reset_once_per_horizon(prepared, candidate, contract, twin):
    ctl = RecordingController((1, 2, 0))
    kwargs = dict(candidate=candidate, contract=contract, schema="instant", seed=456)
    a = evaluate(prepared.horizons["validation"], ctl, twin, **kwargs)
    b = evaluate(prepared.horizons["validation"], ctl, twin, **kwargs)
    assert a == b
    assert ctl.resets == [456, 456]
    assert a.summary["reconfigurations"] == (a.summary["type_switches"]
        + a.summary["scale_up_events"] + a.summary["scale_down_events"])
    assert a.summary["energy_wh"] == pytest.approx(sum(s.outcome.power_w * .25 for s in a.steps))


def test_comparisons_reject_partial_shifted_or_different_workloads(prepared, candidate, contract, twin):
    result = evaluate(prepared.horizons["validation"], RecordingController(), twin,
                      candidate=candidate, contract=contract, schema="instant")
    assert require_comparable((result, result)) == result.horizon.identity
    shifted = replace(result.horizon, partition=replace(result.horizon.partition, start=16))
    other_workload = replace(result.horizon, partition=replace(result.horizon.partition,
        timeline=replace(result.horizon.partition.timeline, load_mbps=(999.0,) * 30)))
    for bad in (replace(result, steps=result.steps[:-1]), replace(result, horizon=shifted),
                replace(result, horizon=other_workload),
                replace(result, contract=replace(contract, cooldown_slots=2))):
        with pytest.raises(ValueError, match="Comparison"):
            require_comparable((result, bad))
