"""Strict interval arithmetic. Candidate ratios are NOT authenticated labels."""
import math


def difference(previous, current, previous_time, current_time, *, sample_seconds=3):
    if previous is None or previous_time is None:return None,'initial_or_missing_previous'
    if current is None or current_time is None:return None,'missing_current'
    if not all(math.isfinite(v) and v>=0 for v in [previous,current]):return None,'invalid_counter'
    if (current_time-previous_time).total_seconds()!=sample_seconds:return None,'gap_or_clock_mismatch'
    if current<previous:return None,'counter_decrease_or_reset'
    return current-previous,'valid'


def candidate_fraction(lost_delta, expected_delta):
    if lost_delta is None or expected_delta is None:return None,'unknown_count'
    if expected_delta<=0:return None,'zero_or_negative_expected_population'
    if lost_delta<0 or lost_delta>expected_delta:return None,'count_inconsistency'
    return lost_delta/expected_delta,'numeric_only_population_unverified'


def authenticate_population(*, definition, source, same_direction, same_interval,
                            same_population, reset_rule_verified):
    if not definition or not source or not all([same_direction,same_interval,same_population,reset_rule_verified]):
        raise ValueError('Loss population definition/source and aligned interval/direction/reset semantics required')
    return dict(definition=definition,source=source,authenticated=True)
