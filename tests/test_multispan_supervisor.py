import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import run_p37_multispan_ab as ab


def fixture_run(root):
    (root / 'inputs').mkdir(parents=True)
    ids = list(range(285)) + [17831]
    protocol = {'expected_batch_plan_sha256': 'order', 'inputs_sha256': {'epoch15.pth': 'initial'},
                'arms': {arm: {'route_increment_weight': 0., 'route_multispan_weight': 0. if arm == 'A_cfm' else .01}
                         for arm in ab.ARMS}, 'comparison_reports': {'old277': 'old.json'}}
    ab.write_json(root / 'protocol.json', protocol)
    ab.write_json(root / 'inputs/source_report.json', {'tasks': [{'task_id': i} for i in ids]})
    np.savez(root / 'inputs/source_latents.npz', **{str(i): np.zeros((1, 4, 8)) for i in ids})
    for arm in ab.ARMS:
        run = root / arm
        (run / 'evaluation/fields').mkdir(parents=True)
        (run / 'checkpoints').mkdir()
        (run / 'checkpoints/best_p37.pth').write_bytes(arm.encode())
        digest = ab.sha(run / 'checkpoints/best_p37.pth')
        ab.write_json(run / 'history.json', [{'train_tasks': 100440, 'validation_tasks': 11160, 'optimizer_steps': 28853}])
        manifest = {key: 'same' for key in ('split_sha256', 'learning_rate', 'batch_size',
            'max_pixels_per_batch', 'batch_plan_epoch_offset', 'source_sha256', 'data_sha256')}
        manifest.update(status='complete', source_checkpoint_sha256='initial', fresh_optimizer=True,
            holdout_used=False, rl=False, cursor={'batch_plan_sha256': 'order'}, route_increment_weight=0.,
            route_multispan_settings={'route_multispan_weight': protocol['arms'][arm]['route_multispan_weight']})
        ab.write_json(run / 'run_manifest.json', manifest)
        fields = np.zeros((1, 4, 2, 2, 2), np.float32)
        field_hash = hashlib.sha256(fields.tobytes()).hexdigest()
        rows = []
        for i in ids:
            np.savez(run / 'evaluation/fields' / f'task_{i:06d}.npz', fields=fields,
                     fields_sha256=np.asarray(field_hash), checkpoint_sha256=np.asarray(digest),
                     sample_network_calls=np.asarray(100), latent=np.zeros((1, 4, 8)))
            rows.append({'task_id': i, 'shape': '768x512',
                         'deployment_success': i < 274 if arm == 'A_cfm' else i < 273 or i == 275,
                         'fields_sha256': field_hash, 'accepted_unsafe_candidates': 0,
                         'reference_continuity_diagnostics': {'candidates': [{'max_cumulative_progress_drawdown': .01}] * 4}})
        ab.write_json(run / 'evaluation/report.json', {'manifest': {'status': 'complete',
            'network_calls_total_bound_to_fields': 28600, 'binding': {
            'checkpoint_sha256': digest, 'seed': 20461115, 'K': 4, 'steps': 100, 'decoder': 'adaptive_fallback/smooth7/512'}},
            'tasks': rows, 'summaries': {'overall': {'success': 274, 'accepted_unsafe_candidates': 0}, 'large': {'success': 274}}})
        ab.write_json(root / 'inputs/old.json', {'tasks': rows})


def test_comparison_reports_gains_losses_and_rejects_changed_latent(tmp_path):
    fixture_run(tmp_path)
    result = ab.compare(tmp_path)
    assert result['paired_changes']['B_dense_multispan_vs_A_cfm'] == {'gains': [275], 'losses': [273]}
    assert not result['validation_losses_directly_comparable']
    path = tmp_path / 'B_dense_multispan/evaluation/fields/task_000001.npz'
    with np.load(path) as cached:
        values = dict(cached)
    values['latent'] = np.ones((1, 4, 8))
    np.savez(path, **values)
    with pytest.raises(AssertionError):
        ab.compare(tmp_path)


def test_comparison_rejects_missing_full_corpus_budget(tmp_path):
    fixture_run(tmp_path)
    path = tmp_path / 'B_dense_multispan/history.json'
    history = json.loads(path.read_text())
    history[0]['train_tasks'] = 100439
    ab.write_json(path, history)
    with pytest.raises(AssertionError):
        ab.compare(tmp_path)


def test_backup_verifies_content_and_retains_checkpoint_for_recovery(tmp_path):
    experiment = tmp_path / 'local_training/pilot'
    path = experiment / 'B_dense_multispan/checkpoints/latest_training_state.pth'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'complete optimizer and RNG recovery fixture')
    receipt = ab.backup(experiment)
    assert receipt['verified_file_contents'] == 1
    with tarfile.open(receipt['archive']) as archive:
        assert archive.extractfile('pilot/B_dense_multispan/checkpoints/latest_training_state.pth').read() == path.read_bytes()
