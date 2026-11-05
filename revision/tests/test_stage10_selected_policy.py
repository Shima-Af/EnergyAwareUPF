"""Selected study policy only; no comparison or scientific fitting."""
import pytest
from revision.stage10.selected_policy import selected_qos,require_authenticated_targets


def case(**changes):
    row=dict(received_mbps=10.,offered_mbps=10.,delay_ms=0.,jitter_us=0.,loss_fraction=0.)
    row.update(changes);return selected_qos(**row)


def test_approved_boundary_and_no_old_weighted_score():
    r=case(received_mbps=9.,delay_ms=4.,jitter_us=1.,loss_fraction=.005)
    assert r['qos']==pytest.approx(.9) and r['compliant'] and r['score_compliant']
    assert r['severity']==0


@pytest.mark.parametrize('field,value',[('received_mbps',8.),('delay_ms',8.),('jitter_us',2.),('loss_fraction',.01)])
def test_each_failure_is_noncompensatory(field,value):
    r=case(**{field:value})
    assert not r['compliant'] and not r['score_compliant']
    assert r['qos']==min(r['components']) and r['severity']>0


def test_bad_historical_loss_provenance_cannot_authorize_training():
    with pytest.raises(ValueError,match='loss_denominator_verified'):
        require_authenticated_targets({'throughput_direction_verified':True})


@pytest.mark.parametrize('kwargs',[{'offered_mbps':0},{'loss_fraction':1.1},{'jitter_us':float('nan')}])
def test_no_missing_or_invalid_kpi_repair(kwargs):
    with pytest.raises(ValueError):case(**kwargs)


def test_loss_reconstruction_does_not_authorize_physical_fraction():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    data=json.loads((root/'revision/reports/stage10/reconciliation_v2/loss_reconstruction.json').read_text())
    for mode in ['dpdk','usr']:
        d=data[mode]
        assert d['matches_at_1e_12']==d['n']
        assert d['stored_RAN_Tx_equals_raw_packet_rate']==d['rate_comparable']
        assert d['raw_rate_header']=='GTPu Packets Tx/s'
        assert not d['physical_ratio_authenticated']


def test_policy_records_exact_author_selected_boundaries():
    import json
    from pathlib import Path
    p=Path(__file__).resolve().parents[1]/'stage10/policy_v1.json'
    d=json.loads(p.read_text());c=d['components']
    assert d['target']==.9 and c['throughput']['minimum_fraction']==.9
    assert c['delay']['budget_ms']==4 and c['jitter']['budget_us']==1
    assert c['loss']['budget_fraction']==.005 and not d['loss_count_substitution_allowed']
