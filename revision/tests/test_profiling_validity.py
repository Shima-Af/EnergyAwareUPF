"""Checks for the read-only validity census, without refitting or model changes."""

import pytest

from revision.diagnostics.profiling_validity import allowed_partition, summarize_failures
from revision.evaluation.twin import RawPrediction


def raw(power, qos):
    return RawPrediction(1, 10.0, 1, 10.0, 10.0, power, power, qos)


def test_invalid_frequency_counts_union_and_keeps_below_target_qos_valid():
    result = summarize_failures([raw(.2, .89), raw(-.02, 1.1), raw(float('nan'), float('nan'))])
    assert result['n'] == 3
    assert result['invalid'] == 2  # Not the sum of overlapping failure causes.
    assert result['invalid_fraction'] == pytest.approx(2 / 3)
    assert result['negative_power'] == result['nonfinite_power'] == 1
    assert result['qos_out_of_range'] == result['nonfinite_qos'] == 1


def test_empty_validity_denominator_fails():
    with pytest.raises(ValueError, match='nonempty'):
        summarize_failures([])


@pytest.mark.parametrize('role', ['train', 'validation'])
def test_only_training_and_validation_inputs_are_authorized(role):
    allowed_partition(role)


@pytest.mark.parametrize('role', ['final_test', 'test', 'all'])
def test_final_or_ambiguous_partition_cannot_enter_census(role):
    with pytest.raises(ValueError, match='no final-test inference'):
        allowed_partition(role)
