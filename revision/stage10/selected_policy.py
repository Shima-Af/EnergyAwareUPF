"""The single user-selected policy. No physical meanings inferred from labels."""
import math
from .primitives import Budgets, KPIs, compliance

APPROVED_BUDGETS=Budgets(.9,4.,1.,.005,'fraction',None)


def selected_qos(*, received_mbps,offered_mbps,delay_ms,jitter_us,loss_fraction):
    values=(received_mbps,offered_mbps,delay_ms,jitter_us,loss_fraction)
    if any(not math.isfinite(v) or v<0 for v in values) or offered_mbps<=0 or loss_fraction>1:
        raise ValueError('Positive offered load and finite physical KPI quantities required')
    ratio=received_mbps/offered_mbps
    # Component saturation is the expressly selected historical normalization,
    # not post-hoc repair of surrogate outputs (invalid values fail above).
    scores=(min(1.,ratio),1-min(delay_ms,40.)/40.,
            1-min(jitter_us,10.)/10.,1-min(100*loss_fraction,5.)/5.)
    result=compliance(KPIs(ratio,delay_ms,jitter_us,loss_fraction),APPROVED_BUDGETS)
    return dict(components=scores,qos=min(scores),score_target=.9,
                score_compliant=min(scores)>=.9,**result)


def require_authenticated_targets(provenance):
    """Fail closed rather than train against a numerically reconstructed bad ratio."""
    required=('throughput_direction_verified','throughput_offered_mapping_verified',
              'loss_numerator_verified','loss_denominator_verified',
              'loss_timebase_verified','loss_direction_verified')
    missing=[k for k in required if provenance.get(k) is not True]
    if missing:raise ValueError('Unauthenticated KPI training targets: '+', '.join(missing))
