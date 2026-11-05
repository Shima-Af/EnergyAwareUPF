"""Shared decision/actuation/accounting contract; no policy or twin is loaded."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, TYPE_CHECKING

from .data import Calendar, DecisionContext, EvaluationHorizon, FitProvenance, fingerprint

if TYPE_CHECKING:
    from .selection import Candidate, FrozenSelection

Schema = Literal["instant", "history", "forecast", "hybrid"]
ACTIONS = (0, 1, 2)  # DPDK, USR(1), USR(2)


@dataclass(frozen=True)
class Contract:
    cooldown_slots: int = 4
    initial_action: int = 0
    qos_target: float = 0.90
    sec_epsilon_mbps: float = 1e-6
    sec_weight: float = 100.0
    qos_weight: float = 30.0
    switch_to_dpdk_cost: float = 0.030
    usr_launch_cost: float = 0.012
    usr_scale_down_cost: float = 0.003
    gamma: float = 0.995

    def __post_init__(self) -> None:
        if type(self.cooldown_slots) is not int or self.cooldown_slots < 0:
            raise ValueError("Cooldown must be a nonnegative integer")
        if type(self.initial_action) is not int or self.initial_action not in ACTIONS:
            raise ValueError("Invalid initial action")
        if not 0 <= self.qos_target <= 1 or not 0 < self.gamma <= 1:
            raise ValueError("Invalid QoS target or discount")
        if not math.isfinite(self.sec_epsilon_mbps) or self.sec_epsilon_mbps <= 0:
            raise ValueError("SEC epsilon must be finite and positive")
        weights = (self.sec_weight, self.qos_weight, self.switch_to_dpdk_cost,
                   self.usr_launch_cost, self.usr_scale_down_cost)
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Reward coefficients must be finite and nonnegative")


@dataclass(frozen=True)
class State:
    action: int
    elapsed_slots: int  # At this decision epoch, before checking the guard.


@dataclass(frozen=True)
class Transition:
    previous: int
    requested: int
    executed: int
    blocked: bool
    type_switch: bool
    scale_up: bool
    scale_down: bool
    usr_instances_launched: int
    next_state: State


def actuate(state: State, requested: int, contract: Contract) -> Transition:
    if type(requested) is not int or requested not in ACTIONS:
        raise ValueError("Action must be an integer in {0,1,2}")
    if state.action not in ACTIONS or state.elapsed_slots < 0:
        raise ValueError("Invalid actuation state")
    changing = requested != state.action
    blocked = changing and state.elapsed_slots < contract.cooldown_slots
    executed = state.action if blocked else requested
    changed = executed != state.action
    type_switch = changed and (executed == 0 or state.action == 0)
    up = changed and executed > state.action > 0
    down = changed and 0 < executed < state.action
    launched = executed if state.action == 0 else max(0, executed - state.action)
    # Next decision is one interval later. A change at t permits another at t+I_c.
    elapsed = min(contract.cooldown_slots, 1 if changed else state.elapsed_slots + 1)
    return Transition(state.action, requested, executed, blocked, type_switch, up, down,
                      launched, State(executed, elapsed))


@dataclass(frozen=True)
class Outcome:
    power_w: float
    qos: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.power_w) or self.power_w < 0:
            raise ValueError("Power must be finite and nonnegative W")
        if not math.isfinite(self.qos) or not 0 <= self.qos <= 1:
            raise ValueError("QoS must be a finite score in [0,1]; no silent clipping")


@dataclass(frozen=True)
class Forecast:
    issued_at: datetime
    observed_through: datetime
    target_at: datetime
    load_mbps: float


class Forecaster(Protocol):
    fit: FitProvenance | None

    def predict(self, context: DecisionContext) -> Forecast: ...


class PersistenceForecast:
    """Causal, parameter-free fixture; not the selected scientific predictor."""

    fit = None

    def predict(self, context: DecisionContext) -> Forecast:
        return Forecast(context.decision_at, context.completed_interval_at,
                        context.decision_at, context.forecast_input_mbps[-1])


def causal_forecast(context: DecisionContext, model: Forecaster,
                    expected_fit: FitProvenance | None) -> Forecast:
    if model.fit != expected_fit:
        raise ValueError("Forecast fitting provenance does not match the declared training data")
    if model.fit is not None and model.fit.last_training_label > context.completed_interval_at:
        raise ValueError("Forecast parameters use labels not available at this decision")
    value = model.predict(context)
    if (value.issued_at != context.decision_at or value.target_at != context.decision_at
            or value.observed_through != context.completed_interval_at):
        raise ValueError("Forecast issuance, target, or observation cutoff is misaligned")
    if not math.isfinite(value.load_mbps) or value.load_mbps < 0:
        raise ValueError("Forecast must be finite, nonnegative Mbps")
    return value


@dataclass(frozen=True)
class Observation:
    decision_at: datetime
    completed_interval_at: datetime
    decision_calendar: Calendar
    completed_calendar: Calendar
    schema: Schema
    current_load_mbps: float  # Most recently completed interval, never the target load.
    history_mbps: tuple[float, ...]
    forecast: Forecast | None
    previous_outcome: Outcome | None  # None at reset; vector adapters must provide a mask.
    state: State
    available_actions: tuple[int, ...] = ACTIONS


def observe(context: DecisionContext, schema: Schema, state: State,
            previous: Outcome | None, forecast: Forecast | None) -> Observation:
    if schema not in ("instant", "history", "forecast", "hybrid"):
        raise ValueError("Unknown observation schema")
    if schema in ("forecast", "hybrid") and forecast is None:
        raise ValueError("Forecast schemas require an aligned causal forecast")
    # All schemas augment the instantaneous signals described in the submission.
    # Do not pass the full context: instant/forecast controllers must not receive
    # the extra history reserved for the history/hybrid observation ablation.
    return Observation(context.decision_at, context.completed_interval_at,
                       context.decision_calendar, context.completed_calendar,
                       schema, context.history_mbps[-1],
                       context.history_mbps if schema in ("history", "hybrid") else (),
                       forecast if schema in ("forecast", "hybrid") else None,
                       previous, state)


class Controller(Protocol):
    """Adapters own recurrent state; reset once per horizon, retain it between calls."""

    def reset(self, *, seed: int) -> None: ...

    def predict(self, observation: Observation, *, deterministic: bool) -> int: ...


class Twin(Protocol):
    """Must be a deterministic, side-effect-free mapping for a fixed artifact bundle."""

    def outcome(self, action: int, load_mbps: float) -> Outcome: ...


@dataclass(frozen=True)
class Accounting:
    energy_wh: float
    offered_mbit: float
    sec_j_per_mbit: float
    qos_shortfall: float
    qos_violation: bool
    energy_term: float
    qos_term: float
    orchestration_term: float
    reward: float


def account(outcome: Outcome, load_mbps: float, slot_seconds: float,
            transition: Transition, contract: Contract) -> Accounting:
    if not math.isfinite(load_mbps) or load_mbps < 0:
        raise ValueError("Load must be finite and nonnegative")
    if not math.isfinite(slot_seconds) or slot_seconds <= 0:
        raise ValueError("Slot duration must be finite and positive")
    sec = outcome.power_w / max(load_mbps, contract.sec_epsilon_mbps)
    shortfall = max(0.0, contract.qos_target - outcome.qos)
    e_term, q_term = contract.sec_weight * sec, contract.qos_weight * shortfall
    orch = (contract.switch_to_dpdk_cost * (transition.type_switch and transition.executed == 0)
            + contract.usr_launch_cost * transition.usr_instances_launched
            + contract.usr_scale_down_cost * transition.scale_down)
    return Accounting(outcome.power_w * slot_seconds / 3600, load_mbps * slot_seconds,
                      sec, shortfall, outcome.qos < contract.qos_target,
                      e_term, q_term, orch, -e_term - q_term - orch)


@dataclass(frozen=True)
class Step:
    index: int
    context: DecisionContext
    forecast: Forecast | None  # Logged before action/step; no post-step lookup.
    load_mbps: float
    transition: Transition
    outcome: Outcome
    accounting: Accounting


@dataclass(frozen=True)
class EvaluationResult:
    candidate_id: str
    horizon: EvaluationHorizon
    contract: Contract
    schema: Schema
    steps: tuple[Step, ...]

    @property
    def summary(self) -> dict[str, float | int | None]:
        energy = math.fsum(s.accounting.energy_wh for s in self.steps)
        volume = math.fsum(s.accounting.offered_mbit for s in self.steps)
        return {
            "slots": len(self.steps),
            "energy_wh": energy,
            "offered_mbit": volume,
            "mean_sec_j_per_mbit": math.fsum(s.accounting.sec_j_per_mbit for s in self.steps) / len(self.steps),
            "aggregate_sec_j_per_mbit": energy * 3600 / volume if volume else None,
            "qos_violation_rate": sum(s.accounting.qos_violation for s in self.steps) / len(self.steps),
            "mean_qos_shortfall": math.fsum(s.accounting.qos_shortfall for s in self.steps) / len(self.steps),
            "total_reward": math.fsum(s.accounting.reward for s in self.steps),
            "discounted_return": math.fsum(self.contract.gamma ** i * s.accounting.reward
                                           for i, s in enumerate(self.steps)),
            "type_switches": sum(s.transition.type_switch for s in self.steps),
            "scale_up_events": sum(s.transition.scale_up for s in self.steps),
            "scale_down_events": sum(s.transition.scale_down for s in self.steps),
            "reconfigurations": sum(s.transition.previous != s.transition.executed for s in self.steps),
            "usr_instances_launched": sum(s.transition.usr_instances_launched for s in self.steps),
            "blocked_requests": sum(s.transition.blocked for s in self.steps),
        }


def require_comparable(results: tuple[EvaluationResult, ...]) -> str:
    """Reject comparison tables with different workloads, horizons, or contracts."""
    if not results:
        raise ValueError("A comparison needs at least one result")
    common = results[0]
    for result in results:
        if (result.horizon.identity != common.horizon.identity
                or result.contract != common.contract
                or tuple(s.index for s in result.steps) != tuple(result.horizon.indices)):
            raise ValueError("Comparison requires identical workloads, full timestamps, and contracts")
    return common.horizon.identity


def evaluate(horizon: EvaluationHorizon, controller: Controller, twin: Twin,
             *, candidate: Candidate, contract: Contract, schema: Schema,
             forecaster: Forecaster | None = None, forecast_fit: FitProvenance | None = None,
             seed: int = 0, frozen_selection: FrozenSelection | None = None) -> EvaluationResult:
    """Generic runner. Final test requires a frozen, validation-selected candidate.

    New learned-forecast training trajectories need causal/walk-forward forecasts;
    a model fitted to the whole training prefix is intentionally rejected there.
    """
    if horizon.partition.timeline.usage == "transfer" and horizon.partition.role != "final_test":
        raise ValueError("Transfer workload is reserved for final reporting only")
    if horizon.partition.role == "final_test":
        if frozen_selection is None:
            raise ValueError("Final testing requires frozen validation-only selection")
        frozen_selection.authorize(candidate, horizon)
        if fingerprint(contract.__dict__) != frozen_selection.contract_id:
            raise ValueError("Final-test contract differs from validation selection")
        if schema != frozen_selection.schema:
            raise ValueError("Final-test observation schema differs from validation selection")
    controller.reset(seed=seed)
    state = State(contract.initial_action, contract.cooldown_slots)
    previous = None
    steps = []
    for index in horizon.indices:
        context = horizon.context(index)
        forecast = causal_forecast(context, forecaster, forecast_fit) if forecaster is not None else None
        observation = observe(context, schema, state, previous, forecast)
        requested = controller.predict(observation, deterministic=True)
        transition = actuate(state, requested, contract)
        load = horizon.partition.timeline.load_mbps[index]
        outcome = twin.outcome(transition.executed, load)
        accounting = account(outcome, load, horizon.partition.timeline.slot.total_seconds(),
                             transition, contract)
        steps.append(Step(index, context, observation.forecast, load, transition, outcome, accounting))
        state, previous = transition.next_state, outcome
    return EvaluationResult(candidate.identity, horizon, contract, schema, tuple(steps))
