"""Tests of scoring reconstruction only; no neural surrogate training."""

import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from revision.diagnostics.reconstruct_qos import NOTEBOOK, DATA, KEYS, components, differences, notebook_functions

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def functions():return notebook_functions(json.loads(NOTEBOOK.read_text()))


@pytest.mark.parametrize('prefix',['dpdk','oai'])
def test_default_schema_really_fails_without_explicit_mapping(functions,prefix):
    frame=pd.read_csv(DATA/f'{prefix}_processed.csv')
    with pytest.raises(KeyError,match='Avg DL Throughput'):
        functions['compute_component_scores'](frame)


def test_four_components_preserve_notebook_clipping_caps_and_weights(functions):
    frame=pd.DataFrame({'DN_Rx_Mbits/s':[0.,20.], 'Objective Throughput (DL/UL)':[10.,10.],
       'downlink_delay_ms':[-1.,80.], 'RAN_jitter':[0.,20.], 'RAN_PLR_interval':[0.,.1]})
    S,_=components(frame,functions)
    np.testing.assert_array_equal(S,[[0.,1.,1.,1.],[1.,0.,0.,0.]])
    np.testing.assert_allclose(functions['weighted_qos'](S),[.6,.4],rtol=0,atol=1e-15)


@pytest.mark.parametrize('prefix',['dpdk','oai'])
def test_every_saved_weighted_label_is_reconstructed(functions,prefix):
    frame=pd.read_csv(DATA/f'{prefix}_processed.csv')
    S,_=components(frame,functions)
    predicted=functions['weighted_qos'](S)
    np.testing.assert_allclose(predicted,frame.QoS_w,atol=2e-16,rtol=0)
    assert not frame.duplicated(KEYS).any()


@pytest.mark.parametrize('prefix',['dpdk','oai'])
def test_pca_reproduces_dd_rather_than_being_the_weighted_target(functions,prefix):
    frame=pd.read_csv(DATA/f'{prefix}_processed.csv')
    S,_=components(frame,functions)
    predicted,artifacts=functions['data_driven_qos'](S)
    np.testing.assert_allclose(predicted,frame.QoS_dd,atol=1e-10,rtol=0)
    assert np.mean(abs(predicted-frame.QoS_w))>.04
    assert int(artifacts['scaler_z'].n_samples_seen_)==len(frame)
    assert artifacts['pca'].n_components_==1
    assert predicted.min()>=-1e-15 and predicted.max()<=1+1e-15


def test_recovered_notebook_does_not_document_calls_to_score_functions():
    notebook=json.loads(NOTEBOOK.read_text())
    counts=[c['execution_count'] for c in notebook['cells'] if c['cell_type']=='code']
    assert counts==[2,25,26,14,24,16,22]
    for c in notebook['cells']:
        if c['cell_type']!='code':continue
        tree=ast.parse(''.join(c['source']))
        for statement in tree.body:
            if isinstance(statement,ast.FunctionDef):continue
            for node in ast.walk(statement):
                assert not (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                     and node.func.id in {'compute_component_scores','weighted_qos','data_driven_qos'})


def test_throughput_explains_previously_unexplained_dl_kpi_label_spans():
    rows=json.loads((ROOT/'revision/reports/task2f/omitted_throughput_explanation.json').read_text())
    assert [r['previous_same_DL_KPI_discrepancy_groups'] for r in rows]==[544,392]
    assert all(r['maximum_within_group_label_span_after_subtracting_throughput_component']<1e-12 for r in rows)


def test_lineage_distinguishes_missing_stage_from_identical_training_copies():
    data=json.loads((ROOT/'revision/reports/task2f/data_lineage.json').read_text())
    assert data['prelabel_files_not_recovered']
    assert not data['full_prelabel_to_training_row_join_available']
    assert all(len(cols)==28 and 'QoS_w' not in cols and 'QoS_dd' not in cols for cols in data['saved_notebook_columns'])
    assert all(r['all_rows_and_values_byte_identical'] for r in data['training_copy_checks'])


def test_numeric_tolerance_is_distinct_from_bitwise_equality():
    r=differences([1.+2e-16,.5],[1.,.5])
    assert r['bitwise_equal_float_values']==1
    assert r['matches_at_1e-10']==2
