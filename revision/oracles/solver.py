"""Exact finite-horizon optimization of represented per-slot costs.

All binary64 costs are converted to common-denominator integers before DP.
Thus no accumulated rounding or tolerance changes a tie or the optimum. The
reward objective uses the same represented gamma**t * reward terms as reporting.
No fitted model, data loader or controller is imported by this solver.
"""
from dataclasses import dataclass
from fractions import Fraction
import math

from revision.evaluation.contract import Contract, Outcome, State, account, actuate

OBJECTIVES = ('reward_optimal_oracle', 'energy_optimal_oracle',
              'qos_constrained_energy_oracle')


@dataclass(frozen=True)
class Solution:
    objective: str
    feasible: bool
    actions: tuple[int, ...]
    objective_value: float | None
    exact_objective_numerator: int | None
    exact_objective_denominator: int | None
    reconfigurations: int | None


def solve(loads, outcomes, *, objective, actions=(0, 1, 2), contract=Contract(),
          slot_seconds=900.0):
    """outcomes[t][action] is a validated Outcome at actual load[t].

    Lexicographic objectives: primary energy or negative discounted return,
    then fewer executed changes (including the initial departure from DPDK),
    then lexicographically smaller executed-action sequence. Blocked requests
    can be omitted without loss: requesting hold yields the same transition.
    Infeasibility has no trajectory and no fabricated/relaxed objective value.
    """
    if objective not in OBJECTIVES:
        raise ValueError('Unknown objective')
    actions = tuple(actions)
    if (actions != tuple(sorted(set(actions))) or not actions
            or any(type(a) is not int or a not in (0, 1, 2) for a in actions)
            or contract.initial_action not in actions):
        raise ValueError('Actions must be sorted, unique and include initial configuration')
    loads = tuple(loads)
    if not loads or len(loads) != len(outcomes):
        raise ValueError('Nonempty aligned loads and outcomes required')
    if any(not math.isfinite(x) or x < 0 for x in loads):
        raise ValueError('Invalid load')
    if not math.isfinite(slot_seconds) or slot_seconds <= 0:
        raise ValueError('Invalid duration')
    for row in outcomes:
        for a in actions:
            if not isinstance(row[a], Outcome):
                raise ValueError('Validated Outcome required')
    states = tuple(State(a, c) for a in actions for c in range(contract.cooldown_slots + 1))
    transitions = {s: tuple(actuate(s, a, contract) for a in actions
                            if a == s.action or s.elapsed_slots >= contract.cooldown_slots)
                   for s in states}
    costs = {}
    denominator = 1
    for t, load in enumerate(loads):
        for s in states:
            for tr in transitions[s]:
                out = outcomes[t][tr.executed]
                if objective == 'qos_constrained_energy_oracle' and out.qos < contract.qos_target:
                    continue
                value = account(out, load, slot_seconds, tr, contract)
                cost = (-contract.gamma ** t * value.reward
                        if objective == 'reward_optimal_oracle' else value.energy_wh)
                if not math.isfinite(cost):
                    raise ValueError('Nonfinite objective term')
                num, den = cost.as_integer_ratio()
                costs[t, s, tr.executed] = (num, den)
                denominator = max(denominator, den)  # All denominators powers of two.
    integers = {k: n * (denominator // d) for k, (n, d) in costs.items()}
    future = {s: (0, 0) for s in states}
    policy = {}
    for t in reversed(range(len(loads))):
        current = {}
        for s in states:
            candidates = []
            for tr in transitions[s]:
                tail = future.get(tr.next_state)
                key = (t, s, tr.executed)
                if tail is not None and key in integers:
                    candidates.append((integers[key] + tail[0],
                                       int(tr.executed != s.action) + tail[1], tr.executed))
            if candidates:
                best = min(candidates)
                current[s] = best[:2]
                policy[t, s] = best[2]
        future = current
    state = State(contract.initial_action, contract.cooldown_slots)
    if state not in future:
        return Solution(objective, False, (), None, None, None, None)
    total, changes = future[state]
    sequence = []
    for t in range(len(loads)):
        a = policy[t, state]
        sequence.append(a)
        state = actuate(state, a, contract).next_state
    exact = Fraction((-total if objective == 'reward_optimal_oracle' else total), denominator)
    return Solution(objective, True, tuple(sequence), float(exact),
                    exact.numerator, exact.denominator, changes)
