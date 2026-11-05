"""QoS diagnostic arithmetic and evidence semantics; no model fitting."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from revision.diagnostics.qos_reliability import LOAD, TARGET, audit_metrics, threshold_statistics, variance_decomposition

ROOT=Path(__file__).resolve().parents[2]


def test_threshold_equality_is_safe_for_measurements_and_predictions():
    result=threshold_statistics([.9,.89,.91,.89],[.9,.9,.89,.89])
    assert result['true_safe']==result['true_unsafe']==result['false_safe']==result['false_unsafe']==1
    assert result['false_safe_fraction_predicted_safe']==.5
    assert result['false_safe_fraction_measured_unsafe']==.5
    assert result['false_safe_fraction_all']==.25


def test_absent_denominators_are_null_not_zero_risk():
    result=threshold_statistics([.95,.96],[.8,.85])
    assert result['false_safe_fraction_predicted_safe'] is None
    assert result['false_safe_fraction_measured_unsafe'] is None
    assert result['false_safe_measured_shortfall'] is None
    empty=threshold_statistics([],[])
    assert empty['n']==0 and empty['false_safe_fraction_all'] is None


def test_false_safe_shortfall_and_optimism_have_distinct_meanings():
    result=threshold_statistics([.5,.8],[.95,.9])
    assert result['false_safe']==2
    assert result['false_safe_measured_shortfall']['max']==pytest.approx(.4)
    assert result['false_safe_measured_shortfall']['mean']==pytest.approx(.25)
    assert result['false_safe_prediction_optimism']['max']==pytest.approx(.45)


@pytest.mark.parametrize('y,p', [([.9],[np.nan]),([np.inf],[.9]),([.9],[.9,.8])])
def test_nonfinite_or_misaligned_inputs_are_not_repaired(y,p):
    with pytest.raises(ValueError):threshold_statistics(y,p)


def test_bias_sign_and_residual_distribution():
    frame=pd.DataFrame({TARGET:[.7,.9], 'predicted_qos':[.8,.9]})
    result=audit_metrics(frame)
    assert result['mae']==pytest.approx(.05)
    assert result['rmse']==pytest.approx(np.sqrt(.005))
    assert result['bias_prediction_minus_measurement']==pytest.approx(.05)
    assert result['residual_quantiles']['0']==0.


def test_same_load_variation_decomposition_is_exact_not_a_trained_model():
    frame=pd.DataFrame({LOAD:[1.,1.,2.,2.],TARGET:[.7,.9,.8,1.],
                        'predicted_qos':[.8,.8,.9,.9],'Test Set':[1,2,1,2]})
    d=variance_decomposition(frame)
    assert d['load_mean_misfit_sse']==pytest.approx(0.,abs=1e-15)
    assert d['model_sse']==pytest.approx(d['within_load_sse'])
    assert d['within_load_fraction_model_sse']==pytest.approx(1.)
    assert d['within_load_case_sse']==0.
    assert d['between_sweeps_given_load_sse']==pytest.approx(d['within_load_sse'])


def test_historical_reproduction_is_not_labeled_independent():
    summary=json.loads((ROOT/'revision/reports/task2e/summary.json').read_text())
    for mode,expected in [('dpdk',(919,41,57)),('usr',(913,52,54))]:
        d=summary[mode];m=d['metrics']['historical_early_stopping_validation']
        assert (m['n'],m['false_safe'],m['false_unsafe'])==expected
        h=d['historical_reconstruction']
        assert h['scaler_statistics']['matches']
        assert h['sweeps_shared']==5 and h['case_blocks_shared']==110
        assert h['median_nearest_same_case_training_seconds']==3.
        assert not h['independent_test']
        assert not d['independent_profiling_evidence_available']


def test_measured_and_predicted_threshold_bands_are_not_conflated():
    rows=json.loads((ROOT/'revision/reports/task2e/near_threshold.json').read_text())
    rows=[r for r in rows if r['mode']=='dpdk' and r['scope']=='historical_early_stopping_validation']
    assert {r['band_anchor']:r['n'] for r in rows}=={'measured':91,'predicted':300}


def test_label_discrepancy_evidence_does_not_claim_authenticated_formula():
    data=json.loads((ROOT/'revision/reports/task2e/label_consistency.json').read_text())
    assert data['groups']['dpdk']['maximum_qos_span']==pytest.approx(.4)
    assert data['groups']['usr']['maximum_qos_span']==pytest.approx(.4)
    assert all(not r['authenticated_processed_KPI_join'] for r in data['raw_export_inventory'])
    provenance=json.loads((ROOT/'revision/reports/task2e/artifact_provenance.json').read_text())
    assert provenance['no_new_model_fitting'] and not provenance['traffic_data_read']
