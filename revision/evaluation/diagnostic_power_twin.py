"""Explicit opt-in USR-power replacement; the default/original twin is untouched."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .contract import Outcome
from .data import fingerprint
from .twin import FittedUPFTwin, InvalidPrediction, RawPrediction, sha256
from revision.profiling.power_model import PowerModel


@dataclass(frozen=True)
class DiagnosticPrediction(RawPrediction):
    power_model_input_mbps: float
    power_source: str


class DiagnosticPowerTwin:
    def __init__(self, original: FittedUPFTwin, model_path: Path, selection_path: Path):
        self.original = original
        self.model_path, self.selection_path = model_path.resolve(), selection_path.resolve()
        selection = json.loads(selection_path.read_text())
        self.model_hash, self.selection_hash = sha256(model_path), sha256(selection_path)
        if self.model_hash != selection['selected_model_sha256']:
            raise ValueError('Replacement model differs from frozen profiling selection')
        self.power_model = PowerModel.load(model_path)
        if self.power_model.name != selection['selected_name'] or selection['traffic_used_for_selection']:
            raise ValueError('Invalid or traffic-selected power replacement')
        self.identity = fingerprint({'original_twin': original.identity, 'power_model': self.model_hash,
                                     'selection': self.selection_hash, 'adapter_sha256': sha256(Path(__file__)),
                                     'power_implementation_sha256': sha256(Path(__file__).parents[1] / 'profiling/power_model.py')})

    def raw_prediction(self, action: int, load_mbps: float) -> DiagnosticPrediction:
        raw = self.original.raw_prediction(action, load_mbps)
        values = asdict(raw)
        if action == 0:
            power_input, source = raw.float32_model_load_mbps, 'unchanged_original_dpdk'
        else:
            power_input = raw.per_instance_load_mbps
            per_power = float(self.power_model.predict([power_input])[0])
            values['per_instance_power_w'], values['power_w'] = per_power, raw.instance_count * per_power
            source = self.power_model.name
        return DiagnosticPrediction(**values, power_model_input_mbps=power_input, power_source=source)

    def outcome(self, action: int, load_mbps: float) -> Outcome:
        raw = self.raw_prediction(action, load_mbps)
        if raw.failures:
            raise InvalidPrediction(raw, self.identity)
        return Outcome(raw.power_w, raw.qos)

    @property
    def manifest(self):
        return {'diagnostic_only': True, 'default_twin_replaced': False, 'twin_id': self.identity,
                'original_twin_id': self.original.identity, 'original_bundle': self.original.manifest,
                'selected_model_path': str(self.model_path), 'selected_model_sha256': self.model_hash,
                'selection_path': str(self.selection_path), 'selection_sha256': self.selection_hash,
                'replacement': 'USR power only; DPDK and all QoS outputs unchanged',
                'legacy_float32_model_load_field_scope': 'Unchanged QoS input; USR power uses explicit float64 power_model_input_mbps',
                'power_model': asdict(self.power_model)}

    def verify_artifacts(self):
        self.original.verify_artifacts()
        if sha256(self.model_path) != self.model_hash or sha256(self.selection_path) != self.selection_hash:
            raise ValueError('Diagnostic replacement artifacts changed after loading')
