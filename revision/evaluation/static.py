"""Deterministic static commands using the common corrected actuation contract."""

from dataclasses import dataclass
from pathlib import Path

from .contract import ACTIONS, Observation
from .data import fingerprint
from .twin import sha256


@dataclass(frozen=True)
class StaticController:
    action: int

    def __post_init__(self) -> None:
        if type(self.action) is not int or self.action not in ACTIONS:
            raise ValueError("Static action must be an integer in {0,1,2}")

    @property
    def identity(self) -> str:
        return fingerprint({"kind": "static", "action": self.action,
                            "implementation_sha256": sha256(Path(__file__))})

    def reset(self, *, seed: int) -> None:
        # No hidden state, random sampling, or native-configuration override.
        if type(seed) is not int:
            raise ValueError("Seed must be an integer")

    def predict(self, observation: Observation, *, deterministic: bool) -> int:
        if deterministic is not True:
            raise ValueError("Static evaluation requires deterministic=True")
        if self.action not in observation.available_actions:
            raise ValueError("Static action is absent from the shared action set")
        return self.action
