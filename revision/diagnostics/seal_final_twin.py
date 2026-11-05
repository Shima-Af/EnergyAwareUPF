"""Check preservation and seal the completed evidence package without Git mutations."""

import json
from pathlib import Path
import re
import subprocess

from revision.evaluation.twin import ROOT, sha256
from revision.profiling.refit_usr_power import write_json


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    out = ROOT / 'revision/reports/final_twin'
    prior_path = out / 'prior_artifacts_sha256.json'
    prior = json.loads(prior_path.read_text())
    changed = [path for path, digest in prior.items()
               if not (ROOT / path).is_file() or sha256(ROOT / path) != digest]
    if changed or git('diff', '--name-only', 'HEAD'):
        raise ValueError(f'Preservation check failed: {changed}')
    assert git('branch', '--show-current') == 'revision/comcom-clean-evaluation'
    assert git('rev-parse', 'HEAD') == 'c0734715418d3f6d10785e5174aebb4bac24d0da'
    historical = json.loads((ROOT / 'revision/configs/twin_artifacts_v1.json').read_text())['artifacts']
    for a in historical:
        assert sha256(ROOT / a['path']) == a['sha256']
    selected = json.loads((ROOT / 'revision/configs/revised_twin_v1.json').read_text())
    for entry in selected['models'].values():
        for a in entry['artifacts']:
            assert sha256(ROOT / a['path']) == a['sha256']
    for a in [selected['qos_specification'], *selected['implementations']]:
        assert sha256(ROOT / a['path']) == a['sha256']
    for filename in ['final_dpdk_power_v1.json', 'usr_power_refit_v1.json']:
        cfg = json.loads((ROOT / 'revision/configs' / filename).read_text())
        assert sha256(Path(cfg['source_csv'])) == cfg['source_sha256']
    report = ROOT / 'revision/FINAL_DIGITAL_TWIN_VALIDATION.md'
    for target in re.findall(r'\]\(([^)]+)\)', report.read_text()):
        if '://' not in target:
            path = report.parent / target.split('#')[0]
            # These two seal files are created below.
            assert path.exists() or path.name in {'file_manifest.json', 'execution_and_integrity.json'}, target
    test_log = ROOT / 'revision/reports/final_twin_tests_full.log'
    assert '218 passed, 28 warnings' in test_log.read_text()
    write_json(out / 'execution_and_integrity.json', {
        'branch': git('branch', '--show-current'), 'head': git('rev-parse', 'HEAD'),
        'tracked_changes_from_head': git('diff', '--name-only', 'HEAD'),
        'status_short': git('status', '--short'),
        'prior_revision_files_checked': len(prior), 'prior_revision_files_changed': changed,
        'prior_snapshot_sha256': sha256(prior_path),
        'original_profiling_artifacts_checked': len(historical),
        'selected_artifact_and_implementation_checks': 'all frozen SHA-256 identities match',
        'profiling_sources': 'both external processed CSV hashes unchanged',
        'full_revision_test_result': {'passed': 218, 'warnings': 28, 'seconds': 22.43,
                                      'log': str(test_log.relative_to(ROOT)), 'sha256': sha256(test_log)},
        'commands': ['.venv/bin/python -m revision.profiling.final_dpdk_power',
                     '.venv/bin/python -m revision.diagnostics.finalize_twin',
                     '.venv/bin/python -m revision.diagnostics.trace_dpdk_power',
                     '.venv/bin/python -m revision.diagnostics.plot_final_twin',
                     '.venv/bin/python -m pytest revision/tests -q -p no:cacheprovider'],
        'execution_environment': 'PYTHONDONTWRITEBYTECODE=1; deterministic CPU runtime from revision.evaluation.twin',
        'scientific_neural_training_this_task': {'dpdk_power_seeds': [42, 123], 'epochs': [143, 135]},
        'initial_failure': 'missing descriptive input_units metadata before any neural training; evidence retained',
        'unit_test_training': 'two preexisting two-epoch synthetic neural fits; no scientific search',
        'controller_training': False, 'traffic_final_test_controller_evaluation': False,
        'manuscript_changes': False, 'git_commit_or_push': False,
        'report_sha256': sha256(report),
        'bytecode_policy': 'incidental __pycache__ files excluded from scientific file manifest; prior recorded artifacts preserved',
    })
    additions = []
    manifest_path = out / 'file_manifest.json'
    for path in sorted((ROOT / 'revision').rglob('*')):
        relative = str(path.relative_to(ROOT))
        if (path.is_file() and '__pycache__' not in path.parts
                and relative not in prior and path != manifest_path):
            additions.append({'path': relative, 'sha256': sha256(path), 'bytes': path.stat().st_size})
    write_json(manifest_path, {'scope': 'all scientific additions in final consolidated twin task',
                              'excludes': ['this self-referential manifest', 'incidental Python bytecode'],
                              'files': additions, 'count': len(additions)})
    print(f'Sealed {len(additions)} additions; {len(prior)} prior files and {len(historical)} original profiling artifacts unchanged')


if __name__ == '__main__':
    main()
