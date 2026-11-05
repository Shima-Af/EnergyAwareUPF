"""Regression tests for the confirmed checkpoint-evaluation random-state defect."""
import json,pickle,random
import numpy as np
import pytest
import revision.stage4
import torch
from stable_baselines3 import PPO
from revision.evaluation.twin import ROOT,sha256
from revision.stage4.rng_isolation import isolated_rng,preserve_rng


def state():return pickle.dumps(random.getstate()),pickle.dumps(np.random.get_state()),torch.get_rng_state().clone()

def same(a,b):return a[0]==b[0] and a[1]==b[1] and torch.equal(a[2],b[2])


def test_real_checkpoint_load_cannot_perturb_training_with_wrapper():
    random.seed(9876);np.random.seed(9876);torch.manual_seed(9876)
    before=state()
    with isolated_rng():PPO.load(ROOT/'revision/models/stage4/controllers/ppo/seed_42/checkpoint_24576/policy.zip',device='cpu')
    assert same(before,state())


def test_rng_stream_next_draws_unchanged_even_on_evaluation_exception():
    before=state()
    expected=(random.random(),float(np.random.rand()),float(torch.rand(1)))
    random.setstate(pickle.loads(before[0]));np.random.set_state(pickle.loads(before[1]));torch.set_rng_state(before[2])
    with pytest.raises(RuntimeError):
        with isolated_rng():
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.rand(20)
            raise RuntimeError('fixture evaluation failure')
    actual=(random.random(),float(np.random.rand()),float(torch.rand(1)))
    assert expected==actual


def test_wrapper_covers_every_checkpoint_capture_and_preserves_function_result():
    @preserve_rng
    def capture():
        random.seed(1);np.random.seed(2);torch.manual_seed(3);return 504
    before=state();assert capture()==504;assert same(before,state())


@pytest.mark.skipif(not (ROOT/'revision/reports/stage4_rng_isolated/controller_comparison.json').exists(),reason='Corrected campaign not complete')
def test_corrected_campaign_full_seed_and_checkpoint_identity():
    from revision.stage4.common import CFG
    base=ROOT/'revision/models/stage4_rng_isolated/controllers'
    for alg in CFG['algorithms']:
        for seed in CFG['seeds']:
            p=base/alg/f'seed_{seed}';selection=json.loads((p/'selection.json').read_text())
            assert selection['actual_timesteps']==300000 and not selection['final_test_used']
            assert [r['timesteps'] for r in selection['validation_scores']]==CFG['checkpoint_timesteps']
            assert selection['selected_checkpoint']==min(selection['validation_scores'],key=lambda r:(-r['total_reward'],r['timesteps']))['checkpoint']
            assert len((p/'training_updates.jsonl').read_text().splitlines())==74
            assert len((p/'episodes.jsonl').read_text().splitlines())==156
            assert json.loads((p/'result.json').read_text())['deterministic_repeat_equal']
            for r in selection['validation_scores']:
                bundle=ROOT/r['checkpoint'];m=json.loads((bundle/'bundle.json').read_text())
                for name,h in m['artifacts'].items():assert sha256(bundle/name)==h
    for name in ['forecasts.csv','persistence_forecasts.csv']:
        assert sha256(ROOT/'revision/reports/stage4_rng_isolated'/name)==sha256(ROOT/'revision/reports/stage4'/name)


@pytest.mark.skipif(not (ROOT/'revision/reports/stage4_rng_isolated/controller_comparison.json').exists(),reason='Corrected campaign not complete')
def test_first_checkpoint_matches_initial_campaign_before_rng_perturbation():
    from revision.stage4.common import CFG
    for alg in CFG['algorithms']:
        for seed in CFG['seeds']:
            name=f'controllers/{alg}/seed_{seed}/checkpoint_24576/validation.json'
            initial=json.loads((ROOT/'revision/models/stage4'/name).read_text())
            corrected=json.loads((ROOT/'revision/models/stage4_rng_isolated'/name).read_text())
            assert initial['metrics']==corrected['metrics']
