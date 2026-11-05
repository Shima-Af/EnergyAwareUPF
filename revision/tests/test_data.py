from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import json

import pytest

from revision.evaluation.data import (
    DataSpec, EvaluationHorizon, FitProvenance, LoadScale, Timeline, TrainingMinMax,
    calendar_at, prepare,
)
from revision.evaluation.inspect_timeline import inspect


def test_split_roles_are_disjoint_and_cover_source(timeline, plan):
    parts = plan.partition(timeline)
    sets = [set(p.timestamps) for p in parts.values()]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]
    assert set.union(*sets) == set(timeline.timestamps)
    assert parts["train"].timestamps[-1] < parts["validation"].timestamps[0]
    assert parts["validation"].timestamps[-1] < parts["final_test"].timestamps[0]


@pytest.mark.parametrize("problem", ["duplicate", "descending", "gap"])
def test_invalid_timestamps_fail_instead_of_sorting_or_filling(timeline, problem):
    times = list(timeline.timestamps)
    if problem == "duplicate":
        times[3] = times[2]
    elif problem == "descending":
        times.reverse()
    else:
        times[3] += timedelta(minutes=1)
    with pytest.raises(ValueError, match="Timestamps"):
        replace(timeline, timestamps=tuple(times))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_missing_or_invalid_loads_fail(timeline, bad):
    with pytest.raises(ValueError, match="Traffic"):
        replace(timeline, load_mbps=(bad, *timeline.load_mbps[1:]))


def test_split_boundaries_must_be_ordered_and_exact(timeline, plan):
    with pytest.raises(ValueError):
        replace(plan, validation_start=plan.final_test_start).partition(timeline)
    with pytest.raises(ValueError, match="exact"):
        replace(plan, validation_start=plan.validation_start + timedelta(minutes=1)).partition(timeline)
    with pytest.raises(ValueError, match="entire"):
        replace(plan, start=timeline.timestamps[1]).partition(timeline)


def test_history_excludes_target_and_preserves_boundary_context(prepared):
    horizon = prepared.horizons["validation"]
    context = horizon.context(15)
    timeline = horizon.partition.timeline
    assert context.history_times == timeline.timestamps[12:15]
    assert context.history_mbps == timeline.load_mbps[12:15]
    assert context.forecast_input_times == timeline.timestamps[11:15]
    assert context.forecast_input_mbps == timeline.load_mbps[11:15]
    assert context.completed_interval_at + timeline.slot == context.decision_at
    assert max(context.forecast_input_times) < context.decision_at
    assert horizon.timestamps == timeline.timestamps[15:23]


def test_common_warmup_once_and_final_row_included(prepared):
    assert tuple(prepared.horizons["train"].indices) == tuple(range(4, 15))
    assert tuple(prepared.horizons["validation"].indices) == tuple(range(15, 23))
    assert tuple(prepared.horizons["final_test"].indices) == tuple(range(23, 30))
    with pytest.raises(ValueError, match="outside"):
        prepared.horizons["validation"].context(14)


def test_calendar_uses_actual_timestamps_including_weekday(prepared):
    context = prepared.horizons["validation"].context(15)
    assert context.decision_calendar == calendar_at(context.decision_at)
    assert context.completed_calendar == calendar_at(context.completed_interval_at)
    assert context.decision_calendar.timestamp != context.completed_calendar.timestamp
    assert calendar_at(context.decision_at - timedelta(days=1)).weekday_cos != context.decision_calendar.weekday_cos


@pytest.mark.parametrize("role", ["validation", "final_test"])
def test_all_fitting_apis_reject_non_training_roles(timeline, plan, role):
    p = plan.partition(timeline)[role]
    with pytest.raises(ValueError, match="training"):
        LoadScale.fit_training(p, 400)
    with pytest.raises(ValueError, match="training"):
        TrainingMinMax.fit_training(p)
    with pytest.raises(ValueError, match="training"):
        FitProvenance.from_training(p)


def test_future_values_do_not_change_fitted_parameters(timeline, plan):
    modified = replace(timeline, load_mbps=timeline.load_mbps[:15] + (1e9,) * 15)
    original_training = plan.partition(timeline)["train"]
    modified_training = plan.partition(modified)["train"]
    a = LoadScale.fit_training(original_training, 400)
    b = LoadScale.fit_training(modified_training, 400)
    assert a.multiplier == b.multiplier
    x = TrainingMinMax.fit_training(original_training)
    y = TrainingMinMax.fit_training(modified_training)
    assert (x.minimum, x.span) == (y.minimum, y.span)
    # Out-of-range validation/test values remain visible; no covert cap/refit.
    assert max(a.transform(modified).load_mbps) > 400
    assert x.transform((1e9,))[0] > 1


def test_constant_training_normalization_is_finite(timeline, plan):
    p = plan.partition(replace(timeline, load_mbps=(2.0,) * 30))["train"]
    normalizer = TrainingMinMax.fit_training(p)
    assert normalizer.transform((2.0, 3.0)) == (0.0, 1.0)


def test_transfer_cannot_fit_and_reuses_source_transforms(timeline, plan, prepared):
    transfer = replace(timeline, name="transfer", usage="transfer", load_mbps=(100.0,) * 30)
    with pytest.raises(ValueError, match="training"):
        LoadScale.fit_training(plan.partition(transfer)["train"], 400)
    with pytest.raises(ValueError, match="reuse"):
        prepare(transfer, plan, DataSpec(3, 4), scale=prepared.load_scale)
    output = prepare(transfer, plan, DataSpec(3, 4), scale=prepared.load_scale,
                     normalizer=prepared.normalizer)
    assert output.normalizer is prepared.normalizer
    assert output.load_scale is prepared.load_scale


def test_csv_reads_original_aggregate_and_ignores_archived_columns(tmp_path):
    p = tmp_path / "traffic.csv"
    p.write_text("timestamp,Traffic_bps,Traffic_Mbps_scaled,Traffic_Predicted_Mbps\n"
                 "2019-04-01 00:00,2000000,999,nan\n"
                 "2019-04-01 00:15,3000000,999,888\n")
    t = Timeline.read_csv(p, "fixture", timedelta(minutes=15))
    assert t.load_mbps == (2.0, 3.0)


def test_actual_repository_timeline_counts_and_aliases():
    root = Path(__file__).resolve().parents[2]
    report = inspect(root / "revision/configs/evaluation_v1.json")
    for dataset in report["datasets"].values():
        assert dataset["rows"] == 3360
        assert [p["label_rows"] for p in dataset["partitions"].values()] == [2016, 504, 840]
        assert [p["evaluation_rows"] for p in dataset["partitions"].values()] == [1920, 504, 840]
        assert dataset["partitions"]["final_test"]["evaluation_last"] == "2019-05-05T23:45:00"
    raw_ids = {r["raw_workload_sha256"] for r in report["archived_csv_inventory"]}
    assert len(raw_ids) == 2  # Five files, two workloads.
    assert all(r["first_archived_forecast_row"] == 96 and not r["archived_forecast_used"]
               for r in report["archived_csv_inventory"])
    assert report == json.loads(json.dumps(report))


def test_longer_window_cannot_silently_drop_a_validation_prefix(prepared):
    # A configured window longer than the available pre-validation context must fail,
    # rather than reducing only some controllers' scored horizon.
    with pytest.raises(ValueError):
        EvaluationHorizon(prepared.partitions["validation"], DataSpec(16, 16))


@pytest.mark.parametrize("spec", [(0, 4, 1), (3, 0, 1), (3.0, 4, 1), (3, 4, 2)])
def test_invalid_windows_and_unsupported_horizon_fail(spec):
    with pytest.raises(ValueError):
        DataSpec(*spec)
