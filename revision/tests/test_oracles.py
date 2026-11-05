"""Synthetic exhaustive checks; never read traffic or fit models."""
from fractions import Fraction
from itertools import product
import math

import numpy as np
import pytest

from revision.evaluation.contract import Contract, Outcome, State, account, actuate
from revision.oracles.solver import OBJECTIVES, solve


def exhaustive(loads, outcomes, actions, objective, contract):
    paths = []
    for requests in product(actions, repeat=len(loads)):
        state = State(contract.initial_action, contract.cooldown_slots)
        total = Fraction(0); changes = 0; sequence = []
        for t, (load, request) in enumerate(zip(loads, requests)):
            tr = actuate(state, request, contract)
            out = outcomes[t][tr.executed]
            if objective == 'qos_constrained_energy_oracle' and out.qos < contract.qos_target:
                break
            value = account(out, load, 900., tr, contract)
            term = (-contract.gamma ** t * value.reward
                    if objective == 'reward_optimal_oracle' else value.energy_wh)
            total += Fraction.from_float(term)
            changes += tr.executed != state.action
            sequence.append(tr.executed); state = tr.next_state
        else:
            paths.append((total, changes, tuple(sequence)))
    return min(paths) if paths else None


@pytest.mark.parametrize('actions', [(0, 1), (0, 1, 2)])
@pytest.mark.parametrize('objective', OBJECTIVES)
@pytest.mark.parametrize('seed', range(4))
def test_exact_dp_matches_all_requested_sequences(actions, objective, seed):
    rng = np.random.default_rng(seed)
    loads = rng.uniform(1, 100, 6)
    outcomes = [[Outcome(float(rng.uniform(0, 3)), float(rng.choice([.85, .9, .95])))
                 for _ in range(3)] for _ in loads]
    contract = Contract()
    oracle = solve(loads, outcomes, actions=actions, objective=objective, contract=contract)
    expected = exhaustive(loads, outcomes, actions, objective, contract)
    assert oracle.feasible == (expected is not None)
    if expected is not None:
        primary = Fraction(oracle.exact_objective_numerator, oracle.exact_objective_denominator)
        if objective == 'reward_optimal_oracle':
            primary = -primary
        assert (primary, oracle.reconfigurations, oracle.actions) == expected


@pytest.mark.parametrize('objective', OBJECTIVES)
def test_equal_objective_prefers_fewer_changes(objective):
    # Remove reward penalties so *all* paths have the same primary objective.
    c = Contract(switch_to_dpdk_cost=0, usr_launch_cost=0, usr_scale_down_cost=0)
    result = solve([1.] * 7, [[Outcome(1, 1)] * 3] * 7, objective=objective, contract=c)
    assert result.actions == (0,) * 7
    assert result.reconfigurations == 0


def test_energy_is_physical_wh_not_reward_or_switch_cost():
    c = Contract(usr_launch_cost=1000)
    outcomes = [[Outcome(4, .99), Outcome(1, .99), Outcome(2, .99)]]
    energy = solve([100.], outcomes, objective='energy_optimal_oracle', contract=c)
    reward = solve([100.], outcomes, objective='reward_optimal_oracle', contract=c)
    assert energy.actions == (1,)
    assert energy.objective_value == .25
    assert reward.actions == (0,)


def test_cooldown_makes_locally_safe_choices_globally_infeasible():
    outcomes = [[Outcome(1, .8), Outcome(1, .95), Outcome(1, .8)],
                [Outcome(1, .95), Outcome(1, .8), Outcome(1, .8)]]
    result = solve([1., 1.], outcomes, objective='qos_constrained_energy_oracle')
    assert not result.feasible
    assert result.actions == () and result.objective_value is None
    assert solve([1., 1.], outcomes, objective='energy_optimal_oracle').feasible


def test_t_plus_four_and_terminal_interval():
    outcomes = [[Outcome(10, 1), Outcome(0, 1), Outcome(20, 1)]]
    outcomes += [[Outcome(0, 1), Outcome(10, 1), Outcome(20, 1)]] * 4
    result = solve([1.] * 5, outcomes, objective='energy_optimal_oracle', actions=(0, 1))
    # Initial switch saves 2.5Wh but forces 7.5Wh later: holding DPDK is optimal.
    assert result.actions == (0,) * 5
    c = Contract(); s = actuate(State(0, 4), 1, c).next_state
    for _ in range(3):
        tr = actuate(s, 0, c); assert tr.blocked; s = tr.next_state
    assert not actuate(s, 0, c).blocked


def test_exact_comparison_does_not_merge_nearby_energy_values():
    row = [Outcome(1, 1), Outcome(math.nextafter(1., 0.), 1), Outcome(2, 1)]
    result = solve([1.], [row], objective='energy_optimal_oracle')
    assert result.actions == (1,)  # Actual energy wins despite extra event.


@pytest.mark.parametrize('objective', OBJECTIVES)
def test_deterministic_and_action_set_bounds(objective):
    loads = [10.] * 5
    outcomes = [[Outcome(3, .95), Outcome(2, .95), Outcome(1, .95)]] * 5
    three = solve(loads, outcomes, objective=objective)
    assert three == solve(loads, outcomes, objective=objective)
    binary = solve(loads, outcomes, objective=objective, actions=(0, 1))
    if objective == 'reward_optimal_oracle':
        assert three.objective_value >= binary.objective_value
    else:
        assert three.objective_value <= binary.objective_value


@pytest.mark.parametrize('kwargs', [dict(actions=(1, 2)), dict(actions=(0, 0)),
                                  dict(objective='energy'), dict(slot_seconds=0)])
def test_invalid_contract_inputs(kwargs):
    args = dict(objective='energy_optimal_oracle'); args.update(kwargs)
    with pytest.raises(ValueError):
        solve([1.], [[Outcome(1, 1)] * 3], **args)
