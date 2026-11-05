from dataclasses import replace

import pytest

from revision.evaluation.contract import evaluate
from revision.evaluation.data import prepare
from revision.evaluation.selection import Candidate, ValidationScore, freeze_selection
from revision.tests.conftest import RecordingController


def validation_result(prepared, candidate, controller, twin, contract):
    return evaluate(prepared.horizons["validation"], controller, twin, candidate=candidate,
                    contract=contract, schema="instant")


@pytest.mark.parametrize("role", ["train", "final_test"])
def test_model_selection_rejects_training_and_final_test_results(prepared, candidate, controller, twin, contract, role):
    result = validation_result(prepared, candidate, controller, twin, contract)
    with pytest.raises(ValueError, match="validation"):
        ValidationScore.from_result(candidate, replace(result, horizon=prepared.horizons[role]), "total_reward")


def test_final_test_unavailable_before_freezing(prepared, candidate, controller, twin, contract):
    with pytest.raises(ValueError, match="frozen"):
        evaluate(prepared.horizons["final_test"], controller, twin, candidate=candidate,
                 contract=contract, schema="instant")
    assert controller.resets == []  # Rejected before running anything.


def test_partial_validation_horizon_cannot_select_a_checkpoint(prepared, candidate, controller, twin, contract):
    result = validation_result(prepared, candidate, controller, twin, contract)
    with pytest.raises(ValueError, match="complete"):
        ValidationScore.from_result(candidate, replace(result, steps=result.steps[:-1]), "total_reward")


def test_frozen_candidate_contract_schema_and_horizon_cannot_change(prepared, candidate, controller, twin, contract):
    result = validation_result(prepared, candidate, controller, twin, contract)
    score = ValidationScore.from_result(candidate, result, "total_reward")
    frozen = freeze_selection((score,), validation=prepared.horizons["validation"],
                              final_test=prepared.horizons["final_test"], objective="total_reward")
    kwargs = dict(candidate=candidate, contract=contract, schema="instant", frozen_selection=frozen)
    with pytest.raises(ValueError, match="candidate or horizon"):
        evaluate(prepared.horizons["final_test"], controller, twin,
                 **{**kwargs, "candidate": Candidate("changed", (("config", "b" * 64),))})
    with pytest.raises(ValueError, match="contract"):
        evaluate(prepared.horizons["final_test"], controller, twin,
                 **{**kwargs, "contract": replace(contract, qos_target=.95)})
    with pytest.raises(ValueError, match="schema"):
        evaluate(prepared.horizons["final_test"], controller, twin, **{**kwargs, "schema": "history"})
    shortened = replace(prepared.horizons["final_test"],
                        partition=replace(prepared.partitions["final_test"], stop=29))
    with pytest.raises(ValueError, match="candidate or horizon"):
        evaluate(shortened, controller, twin, **kwargs)


def test_final_test_values_never_enter_model_selection(prepared, candidate, controller, twin, contract):
    result = validation_result(prepared, candidate, controller, twin, contract)
    score = ValidationScore.from_result(candidate, result, "total_reward")
    worse = replace(score, candidate=Candidate("worse", (("config", "c" * 64),)), value=score.value - 1)
    chosen = freeze_selection((worse, score), validation=prepared.horizons["validation"],
                              final_test=prepared.horizons["final_test"], objective="total_reward")
    assert chosen.candidate == candidate
    # Selection APIs accept validation scores only, not final-test performance.
    assert "final_test_score" not in chosen.__dict__
    assert controller.resets == [0]
    assert max(o.decision_at for o in controller.observations) < prepared.horizons["final_test"].timestamps[0]


def test_selection_rejects_mixed_horizons_and_objectives(prepared, candidate, controller, twin, contract):
    result = validation_result(prepared, candidate, controller, twin, contract)
    score = ValidationScore.from_result(candidate, result, "total_reward")
    kwargs = dict(validation=prepared.horizons["validation"],
                  final_test=prepared.horizons["final_test"], objective="total_reward")
    for bad in [replace(score, horizon_id="other"), replace(score, objective="discounted_return"),
                replace(score, value=float("nan"))]:
        with pytest.raises(ValueError):
            freeze_selection((bad,), **kwargs)
    with pytest.raises(ValueError, match="contract"):
        freeze_selection((score, replace(score, contract_id="other")), **kwargs)


def test_changing_all_final_loads_cannot_change_validation_winner(prepared, candidate, twin, contract):
    original_timeline = prepared.partitions["train"].timeline
    cutoff = prepared.partitions["final_test"].start
    poisoned = replace(original_timeline,
                       load_mbps=original_timeline.load_mbps[:cutoff] + (1e9,) * (30 - cutoff))
    winners, validation_values = [], []
    for timeline in (original_timeline, poisoned):
        val = replace(prepared.horizons["validation"],
                      partition=replace(prepared.partitions["validation"], timeline=timeline))
        final = replace(prepared.horizons["final_test"],
                        partition=replace(prepared.partitions["final_test"], timeline=timeline))
        scores = []
        for identity, actions in [(candidate, (0,)),
                                  (Candidate("second", (("config", "d" * 64),)), (1,))]:
            result = evaluate(val, RecordingController(actions), twin, candidate=identity,
                              contract=contract, schema="instant")
            scores.append(ValidationScore.from_result(identity, result, "total_reward"))
        winners.append(freeze_selection(tuple(scores), validation=val, final_test=final,
                                        objective="total_reward").candidate)
        validation_values.append(tuple(s.value for s in scores))
    assert validation_values[0] == validation_values[1]
    assert winners[0] == winners[1]


def test_transfer_reports_only_after_source_validation_selection(timeline, plan, prepared, candidate, controller, twin, contract):
    transfer = prepare(replace(timeline, name="youtube_fixture", usage="transfer"), plan,
                       prepared.horizons["train"].spec, scale=prepared.load_scale,
                       normalizer=prepared.normalizer)
    with pytest.raises(ValueError, match="reporting"):
        evaluate(transfer.horizons["validation"], controller, twin, candidate=candidate,
                 contract=contract, schema="instant")
    result = validation_result(prepared, candidate, controller, twin, contract)
    with pytest.raises(ValueError, match="validation"):
        ValidationScore.from_result(candidate, replace(result, horizon=transfer.horizons["validation"]), "total_reward")
    score = ValidationScore.from_result(candidate, result, "total_reward")
    frozen = freeze_selection((score,), validation=prepared.horizons["validation"],
                              final_test=prepared.horizons["final_test"], objective="total_reward",
                              transfer_tests=(transfer.horizons["final_test"],))
    output = evaluate(transfer.horizons["final_test"], controller, twin, candidate=candidate,
                      contract=contract, schema="instant", frozen_selection=frozen)
    assert output.horizon.timestamps == prepared.horizons["final_test"].timestamps
