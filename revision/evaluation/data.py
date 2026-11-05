"""Immutable timelines, chronological partitions, and training-only transforms.

CSV timestamps label interval starts. No sorting, filling, resampling, or use of
the historical forecast/scaled columns is permitted by the preparation path.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

Role = Literal["train", "validation", "final_test"]


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Timeline:
    name: str
    timestamps: tuple[datetime, ...]
    load_mbps: tuple[float, ...]
    slot: timedelta
    source_sha256: str
    usage: Literal["development", "transfer"] = "development"

    def __post_init__(self) -> None:
        if not self.timestamps or len(self.timestamps) != len(self.load_mbps):
            raise ValueError("A timeline needs equally sized, nonempty times and loads")
        if self.slot.total_seconds() <= 0:
            raise ValueError("Slot duration must be positive")
        # Preserve the source's naive clock; timezone provenance is unresolved.
        if any(t.tzinfo is not None for t in self.timestamps):
            raise ValueError("Expected timezone-naive source timestamps")
        if any(b - a != self.slot for a, b in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("Timestamps must be unique, ascending, and regularly spaced")
        if any(not math.isfinite(x) or x < 0 for x in self.load_mbps):
            raise ValueError("Traffic must be finite, nonnegative Mbps")
        if self.usage not in ("development", "transfer"):
            raise ValueError("Unknown workload usage")

    @property
    def identity(self) -> str:
        return fingerprint([self.name, self.usage, self.source_sha256, self.slot.total_seconds(),
                            [t.isoformat() for t in self.timestamps], self.load_mbps])

    @classmethod
    def read_csv(cls, path: Path, name: str, slot: timedelta,
                 *, usage: Literal["development", "transfer"] = "development") -> Timeline:
        contents = path.read_bytes()
        rows = list(csv.DictReader(contents.decode("utf-8-sig").splitlines()))
        # Read the pre-rescaling aggregate only. Archived forecasts are ignored.
        return cls(name, tuple(datetime.fromisoformat(r["timestamp"]) for r in rows),
                   tuple(float(r["Traffic_bps"]) / 1e6 for r in rows), slot,
                   hashlib.sha256(contents).hexdigest(), usage)


@dataclass(frozen=True)
class Partition:
    timeline: Timeline
    role: Role
    start: int
    stop: int  # Exclusive; labels/outcomes in [start, stop), context may precede start.

    def __post_init__(self) -> None:
        if self.role not in ("train", "validation", "final_test"):
            raise ValueError("Unknown partition role")
        if not 0 <= self.start < self.stop <= len(self.timeline.timestamps):
            raise ValueError("Invalid partition boundaries")

    @property
    def timestamps(self) -> tuple[datetime, ...]:
        return self.timeline.timestamps[self.start:self.stop]

    @property
    def loads(self) -> tuple[float, ...]:
        return self.timeline.load_mbps[self.start:self.stop]

    @property
    def identity(self) -> str:
        return fingerprint([self.timeline.identity, self.role, self.start, self.stop])

    def require_training(self) -> None:
        if self.role != "train" or self.timeline.usage != "development":
            raise ValueError("Parameter fitting is restricted to training data")


@dataclass(frozen=True)
class SplitPlan:
    start: datetime
    validation_start: datetime
    final_test_start: datetime
    end_exclusive: datetime

    def partition(self, timeline: Timeline) -> dict[Role, Partition]:
        bounds = (self.start, self.validation_start, self.final_test_start, self.end_exclusive)
        if not all(a < b for a, b in zip(bounds, bounds[1:])):
            raise ValueError("Split boundaries must be strictly chronological")
        times = timeline.timestamps
        if self.start != times[0] or self.end_exclusive != times[-1] + timeline.slot:
            raise ValueError("Split plan must cover the entire supplied timeline")
        for boundary in bounds[1:-1]:
            if boundary not in times:
                raise ValueError("Split boundary must be an exact source timestamp")
        indices = [bisect.bisect_left(times, b) for b in bounds]
        return {role: Partition(timeline, role, a, b)
                for role, a, b in zip(("train", "validation", "final_test"), indices, indices[1:])}


@dataclass(frozen=True)
class FitProvenance:
    training_partition_id: str
    last_training_label: datetime

    @classmethod
    def from_training(cls, partition: Partition) -> FitProvenance:
        partition.require_training()
        return cls(partition.identity, partition.timestamps[-1])


@dataclass(frozen=True)
class LoadScale:
    """Workload design: training peak maps to a declared reference Mbps."""

    multiplier: float
    fit: FitProvenance

    @classmethod
    def fit_training(cls, partition: Partition, reference_peak_mbps: float) -> LoadScale:
        provenance = FitProvenance.from_training(partition)
        peak = max(partition.loads)
        if peak <= 0 or not math.isfinite(reference_peak_mbps) or reference_peak_mbps <= 0:
            raise ValueError("Training peak and reference peak must be positive")
        return cls(reference_peak_mbps / peak, provenance)

    def transform(self, timeline: Timeline) -> Timeline:
        # Never refit to validation, test, or a transfer trace; never clip peaks.
        return replace(timeline, load_mbps=tuple(x * self.multiplier for x in timeline.load_mbps))


@dataclass(frozen=True)
class TrainingMinMax:
    minimum: float
    span: float
    fit: FitProvenance

    @classmethod
    def fit_training(cls, partition: Partition) -> TrainingMinMax:
        provenance = FitProvenance.from_training(partition)
        low, high = min(partition.loads), max(partition.loads)
        return cls(low, high - low if high > low else 1.0, provenance)

    def transform(self, values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple((x - self.minimum) / self.span for x in values)


@dataclass(frozen=True)
class Calendar:
    timestamp: datetime
    time_sin: float
    time_cos: float
    weekday_sin: float
    weekday_cos: float


def calendar_at(timestamp: datetime) -> Calendar:
    day_fraction = (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second) / 86400
    return Calendar(timestamp, math.sin(2 * math.pi * day_fraction),
                    math.cos(2 * math.pi * day_fraction),
                    math.sin(2 * math.pi * timestamp.weekday() / 7),
                    math.cos(2 * math.pi * timestamp.weekday() / 7))


@dataclass(frozen=True)
class DataSpec:
    history_slots: int
    forecast_input_slots: int
    forecast_horizon_slots: int = 1

    def __post_init__(self) -> None:
        if any(type(x) is not int for x in (self.history_slots, self.forecast_input_slots,
                                          self.forecast_horizon_slots)):
            raise ValueError("Window sizes and forecast horizon must be integers")
        if self.history_slots < 1 or self.forecast_input_slots < 1:
            raise ValueError("History and forecast windows must be positive")
        if self.forecast_horizon_slots != 1:
            raise ValueError("This protocol implements only the submitted one-step horizon")

    @property
    def warmup(self) -> int:
        return max(self.history_slots, self.forecast_input_slots)


@dataclass(frozen=True)
class DecisionContext:
    decision_at: datetime
    completed_interval_at: datetime
    history_times: tuple[datetime, ...]
    history_mbps: tuple[float, ...]
    forecast_input_times: tuple[datetime, ...]
    forecast_input_mbps: tuple[float, ...]
    decision_calendar: Calendar
    completed_calendar: Calendar


@dataclass(frozen=True)
class EvaluationHorizon:
    partition: Partition
    spec: DataSpec

    def __post_init__(self) -> None:
        if self.partition.role != "train" and self.partition.start < self.spec.warmup:
            raise ValueError("Insufficient preceding context for the fixed evaluation horizon")
        if max(self.partition.start, self.spec.warmup) >= self.partition.stop:
            raise ValueError("No decisions remain after the common warmup")

    @property
    def indices(self) -> range:
        # Common for every controller and observation schema; no split-local burn-in.
        return range(max(self.partition.start, self.spec.warmup), self.partition.stop)

    @property
    def timestamps(self) -> tuple[datetime, ...]:
        return tuple(self.partition.timeline.timestamps[i] for i in self.indices)

    @property
    def identity(self) -> str:
        return fingerprint([self.partition.identity, self.spec.__dict__, list(self.indices)])

    def context(self, index: int) -> DecisionContext:
        if index not in self.indices:
            raise ValueError("Index is outside the common evaluation horizon")
        t, s = self.partition.timeline, self.spec
        return DecisionContext(t.timestamps[index], t.timestamps[index - 1],
                               t.timestamps[index - s.history_slots:index],
                               t.load_mbps[index - s.history_slots:index],
                               t.timestamps[index - s.forecast_input_slots:index],
                               t.load_mbps[index - s.forecast_input_slots:index],
                               calendar_at(t.timestamps[index]), calendar_at(t.timestamps[index - 1]))


@dataclass(frozen=True)
class PreparedData:
    partitions: dict[Role, Partition]
    horizons: dict[Role, EvaluationHorizon]
    load_scale: LoadScale
    normalizer: TrainingMinMax


def prepare(timeline: Timeline, plan: SplitPlan, spec: DataSpec,
            *, scale: LoadScale, normalizer: TrainingMinMax | None = None) -> PreparedData:
    partitions = plan.partition(scale.transform(timeline))
    # For a transfer workload, the Main-fitted normalizer must be passed explicitly.
    if normalizer is None:
        if scale.fit.training_partition_id != plan.partition(timeline)["train"].identity:
            raise ValueError("A transfer trace must reuse the source training normalizer")
        normalizer = TrainingMinMax.fit_training(partitions["train"])
    return PreparedData(partitions, {r: EvaluationHorizon(p, spec) for r, p in partitions.items()},
                        scale, normalizer)
