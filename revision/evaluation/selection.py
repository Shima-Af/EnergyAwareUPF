"""Validation-only selection and a frozen identity gate for final reporting.

These are accidental-leakage guards, not a security boundary against fabricated
metadata. Adapters must hash the actual artifacts they load, including paired
checkpoint/normalization statistics and forecast/preprocessing configuration.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from .contract import EvaluationResult, Schema
from .data import EvaluationHorizon, fingerprint

Objective = Literal["total_reward", "discounted_return", "negative_energy_wh"]


@dataclass(frozen=True)
class Candidate:
    name: str
    artifact_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        names = [name for name, _ in self.artifact_hashes]
        if not self.name or not names or len(names) != len(set(names)):
            raise ValueError("Candidates require named, unique artifact identities")
        if any(not name or re.fullmatch(r"[0-9a-f]{64}", digest) is None
               for name, digest in self.artifact_hashes):
            raise ValueError("Artifacts must have lowercase SHA-256 hashes")

    @property
    def identity(self) -> str:
        return fingerprint([self.name, sorted(self.artifact_hashes)])


@dataclass(frozen=True)
class ValidationScore:
    candidate: Candidate
    horizon_id: str
    objective: Objective
    value: float
    contract_id: str
    schema: Schema

    @classmethod
    def from_result(cls, candidate: Candidate, result: EvaluationResult,
                    objective: Objective) -> ValidationScore:
        if (result.horizon.partition.role != "validation"
                or result.horizon.partition.timeline.usage != "development"):
            raise ValueError("Model selection accepts validation results only")
        if result.candidate_id != candidate.identity:
            raise ValueError("Candidate identity differs from evaluated artifacts")
        if tuple(s.index for s in result.steps) != tuple(result.horizon.indices):
            raise ValueError("Selection requires the complete common validation horizon")
        if objective not in ("total_reward", "discounted_return", "negative_energy_wh"):
            raise ValueError("Unknown prespecified selection objective")
        value = (-result.summary["energy_wh"] if objective == "negative_energy_wh"
                 else result.summary[objective])
        if not math.isfinite(value):
            raise ValueError("Selection scores must be finite")
        return cls(candidate, result.horizon.identity, objective, value,
                   fingerprint(result.contract.__dict__), result.schema)


@dataclass(frozen=True)
class FrozenSelection:
    candidate: Candidate
    validation_horizon_id: str
    final_test_horizon_id: str
    objective: Objective
    contract_id: str
    schema: Schema
    transfer_horizon_ids: tuple[str, ...]

    def authorize(self, candidate: Candidate, horizon: EvaluationHorizon) -> None:
        if (candidate != self.candidate or horizon.partition.role != "final_test"
                or horizon.identity not in (self.final_test_horizon_id, *self.transfer_horizon_ids)):
            raise ValueError("Final test candidate or horizon differs from frozen selection")


def freeze_selection(scores: tuple[ValidationScore, ...], *, validation: EvaluationHorizon,
                     final_test: EvaluationHorizon, objective: Objective,
                     transfer_tests: tuple[EvaluationHorizon, ...] = ()) -> FrozenSelection:
    if validation.partition.role != "validation" or final_test.partition.role != "final_test":
        raise ValueError("Expected validation and final-test horizons")
    if validation.partition.timeline.usage != "development":
        raise ValueError("A transfer workload cannot select models")
    if (validation.partition.timeline.identity != final_test.partition.timeline.identity
            or validation.partition.stop != final_test.partition.start or validation.spec != final_test.spec):
        raise ValueError("Selection and reporting must share a dataset and protocol")
    if not scores or any(s.horizon_id != validation.identity or s.objective != objective for s in scores):
        raise ValueError("All candidates must use the same validation horizon and objective")
    if len({s.contract_id for s in scores}) != 1:
        raise ValueError("Candidates must share the evaluation contract")
    if any(not math.isfinite(s.value) for s in scores):
        raise ValueError("Nonfinite score")
    for transfer in transfer_tests:
        if (transfer.partition.role != "final_test" or transfer.partition.timeline.usage != "transfer"
                or transfer.spec != final_test.spec or transfer.timestamps != final_test.timestamps):
            raise ValueError("Transfer reporting must use the same final timestamps and window contract")
    # Stable tie breaking, independent of trial ordering or final-test behavior.
    best = min(scores, key=lambda s: (-s.value, s.candidate.identity))
    return FrozenSelection(best.candidate, validation.identity, final_test.identity,
                           objective, best.contract_id, best.schema,
                           tuple(t.identity for t in transfer_tests))
