"""Frozen full-corpus A/B supervisor for one dense multispan objective."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tarfile
import traceback

import numpy as np

from run_p37_increment_ab import sha, write_json, local_now

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('A_cfm', 'B_dense_multispan')


def compare(experiment):
    protocol = json.loads((experiment / 'protocol.json').read_text())
    source = json.loads((experiment / 'inputs/source_report.json').read_text())
    ids = [row['task_id'] for row in source['tasks']]
    reports, histories, manifests, checkpoints = {}, {}, {}, {}
    latents = {}
    original_latents = np.load(experiment / 'inputs/source_latents.npz')
    for arm in ARMS:
        run = experiment / arm
        report = reports[arm] = json.loads((run / 'evaluation/report.json').read_text())
        history = histories[arm] = json.loads((run / 'history.json').read_text())
        manifest = manifests[arm] = json.loads((run / 'run_manifest.json').read_text())
        checkpoints[arm] = sha(run / 'checkpoints/best_p37.pth')
        assert manifest['status'] == report['manifest']['status'] == 'complete'
        assert report['manifest']['network_calls_total_bound_to_fields'] == 28600
        assert len(history) == 1
        assert (history[0]['train_tasks'], history[0]['validation_tasks'], history[0]['optimizer_steps']) == (100440, 11160, 28853)
        assert [row['task_id'] for row in report['tasks']] == ids and len(ids) == 286
        binding = report['manifest']['binding']
        assert binding['checkpoint_sha256'] == checkpoints[arm]
        assert (binding['seed'], binding['K'], binding['steps'], binding['decoder']) == (20461115, 4, 100, 'adaptive_fallback/smooth7/512')
        assert manifest['source_checkpoint_sha256'] == protocol['inputs_sha256']['epoch15.pth']
        assert manifest['cursor']['batch_plan_sha256'] == protocol['expected_batch_plan_sha256']
        assert manifest['fresh_optimizer'] and not manifest['holdout_used'] and not manifest['rl']
        for key, expected in protocol['arms'][arm].items():
            if key.startswith('route_multispan_'):
                assert manifest['route_multispan_settings'].get(key, 0 if key.endswith('_weight') else None) == expected
            elif key == 'route_increment_weight':
                assert manifest[key] == expected
        latents[arm] = []
        for row in report['tasks']:
            with np.load(run / 'evaluation/fields' / f"task_{row['task_id']:06d}.npz") as cached:
                assert str(cached['checkpoint_sha256']) == checkpoints[arm]
                assert int(cached['sample_network_calls']) == 100
                field_hash = hashlib.sha256(np.ascontiguousarray(cached['fields']).tobytes()).hexdigest()
                assert field_hash == row['fields_sha256'] == str(cached['fields_sha256'])
                latents[arm].append(cached['latent'])
                assert np.array_equal(cached['latent'], original_latents[str(row['task_id'])])
    for key in ('source_checkpoint_sha256', 'split_sha256', 'learning_rate', 'batch_size',
                'max_pixels_per_batch', 'batch_plan_epoch_offset', 'source_sha256', 'data_sha256'):
        assert manifests[ARMS[0]][key] == manifests[ARMS[1]][key], key
    assert all(np.array_equal(a, b) for a, b in zip(latents[ARMS[0]], latents[ARMS[1]]))
    baseline_reports = {name: json.loads((experiment / 'inputs' / filename).read_text())
                        for name, filename in protocol['comparison_reports'].items()}
    indexed = {name: {r['task_id']: r for r in report['tasks']}
               for name, report in (reports | baseline_reports).items()}
    assert all(set(rows) == set(ids) for rows in indexed.values())
    pairs = {}
    for arm in ARMS:
        for name in (*baseline_reports, *([ARMS[0]] if arm == ARMS[1] else [])):
            pairs[f'{arm}_vs_{name}'] = {
                'gains': [tid for tid in ids if indexed[arm][tid]['deployment_success'] and not indexed[name][tid]['deployment_success']],
                'losses': [tid for tid in ids if not indexed[arm][tid]['deployment_success'] and indexed[name][tid]['deployment_success']],
            }
    mechanism = {}
    for arm, report in reports.items():
        mechanism[arm] = {}
        for group, rows in [('overall', report['tasks']), ('large', [r for r in report['tasks'] if r['shape'] in ('512x768', '768x512')])]:
            candidates = [c for row in rows for c in row['reference_continuity_diagnostics']['candidates']]
            mechanism[arm][group] = {key: {'median': float(np.median([c[key] for c in candidates])),
                                         'p95': float(np.quantile([c[key] for c in candidates], .95)),
                                         'max': float(max(c[key] for c in candidates))}
                                     for key in candidates[0]}
        mechanism[arm]['all_failed_tasks'] = [r for r in report['tasks'] if not r['deployment_success']]
        mechanism[arm]['task17831'] = indexed[arm][17831]
    result = {'status': 'complete', 'finished_local': local_now(),
        'arms': {arm: {'history': histories[arm], 'summaries': reports[arm]['summaries'],
                       'checkpoint_sha256': checkpoints[arm]} for arm in ARMS},
        'paired_changes': pairs, 'mechanism_diagnostics': mechanism,
        'old_cache_mechanism_baselines': 'baseline_continuity.json',
        'equal_batch_order_verified': True, 'identical_latents_verified': True,
        'actual_integrated_fields_hash_verified': True, 'validation_losses_directly_comparable': False,
        'current_model_replaced': False, 'holdout_used': False,
        'limitations': ['One epoch/seed is preliminary; development tasks are reused.',
            'Reference-route diagnostics can mismatch a valid alternate predicted route.',
            'The objective supervises CFM endpoint estimates; actual integrated fields decide geometry.']}
    write_json(experiment / 'comparison.json', result)
    lines = ['# 密采样多跨度进度目标：全样本单轮 A/B', '', f"完成：{result['finished_local']}", '',
        'A：原CFM；B：仅增加0.01倍沿配对路线的多跨度相对进度斜率Smooth-L1。',
        '两组同epoch15初始化、新AdamW、相同种子/批序，各100,440训练与11,160验证。',
        '固定adaptive_fallback/smooth7/512、K4/100步，实际重新采样各286任务。', '',
        '| 组别 | 全部 | 最大两尺寸 | 接受但不安全候选 |', '|---|---:|---:|---:|']
    for arm, report in reports.items():
        whole, large = report['summaries']['overall'], report['summaries']['large']
        lines.append(f"| {arm} | {whole['success']}/286 | {large['success']}/44 | {whole['accepted_unsafe_candidates']} |")
    for name, changes in pairs.items():
        lines += ['', f"{name}：新增{changes['gains']}；丢失{changes['losses']}。"]
    lines += ['', '两种总loss不直接排名。机制指标包含累计回退、低推进窗口和支撑缺口；不能替代几何验收。',
              '单轮单种子只提供初筛证据；不自动延长或替换模型，保留140条未见尺寸任务。',
              '完整逐任务与候选数据见comparison.json、两组evaluation/report.json。']
    (experiment / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return result


def verify_geometry(experiment):
    """Recompute swept-disk geometry from every saved candidate path."""
    sys.path[:0] = [str(ROOT / 'training/source/map_flow_matching_p1'),
                   str(ROOT / 'training/source/map_flow_matching')]
    import torch
    from utils.dataset import PathPlanningDataset
    from scripts.geometry_eval import score_task
    source = json.loads((experiment / 'inputs/source_report.json').read_text())
    base = PathPlanningDataset(source['manifest']['binding']['data_path'], normalize=True,
        augment=False, distance_scale=8., validate_geometry=False, cache_distance_fields=True)
    verified = {}
    for arm in ARMS:
        run = experiment / arm
        state = torch.load(run / 'checkpoints/best_p37.pth', map_location='cpu', weights_only=False)
        assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values())
        report = json.loads((run / 'evaluation/report.json').read_text())
        for row in report['tasks']:
            tid = row['task_id']
            with np.load(run / 'evaluation/paths' / f'task_{tid:06d}.npz') as saved:
                paths = saved['paths_px']
            assert paths.shape == (4, 512, 2) and np.isfinite(paths).all()
            assert np.array_equal(paths[:, 0], np.broadcast_to(base.starts[tid], (4, 2)))
            assert np.array_equal(paths[:, -1], np.broadcast_to(base.goals[tid], (4, 2)))
            safe = np.asarray(score_task((str(ROOT / 'training/source/map_flow_matching'), paths,
                base.maps[tid], base.starts[tid], base.goals[tid], base.geometry.to_json())), dtype=bool)
            assert np.array_equal(safe, row['geometry_safe'])
            accepted = np.asarray(row['accepted'], dtype=bool)
            selected = row['selected_index']
            assert bool(safe[selected] and accepted[selected]) == row['deployment_success']
            assert int((~safe & accepted).sum()) == row['accepted_unsafe_candidates']
        verified[arm] = {'tasks': len(report['tasks']), 'finite_exact_endpoint_candidates': 4 * len(report['tasks']),
                         'geometry_recomputed': True, 'weights_finite': True}
    write_json(experiment / 'geometry_verification.json', {'finished_local': local_now(), 'arms': verified})


def backup(experiment):
    """Validate every archived regular member against a fixed content list."""
    contents = {str(p.relative_to(experiment)): sha(p) for p in experiment.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts and p.name not in ('archive_content_manifest.json', 'backup_receipt.json')}
    write_json(experiment / 'archive_content_manifest.json', contents)
    stamp = local_now().replace('-', '').replace(':', '')[:15].replace('T', '_')
    archive = experiment.parent.parent / 'backups' / f'p37_multispan_ab_{stamp}.tar.gz'
    archive.parent.mkdir(exist_ok=True)
    if archive.exists():
        raise FileExistsError(archive)
    temp = archive.with_suffix('.tmp')
    with tarfile.open(temp, 'w:gz', compresslevel=3) as tar:
        tar.add(experiment, arcname=experiment.name, filter=lambda info: None if '__pycache__' in Path(info.name).parts else info)
    temp.replace(archive)
    verified = 0
    with tarfile.open(archive, 'r:gz') as tar:
        for relative, expected in contents.items():
            digest = hashlib.sha256()
            with tar.extractfile(f'{experiment.name}/{relative}') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            assert digest.hexdigest() == expected, relative
            verified += 1
    receipt = {'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive),
               'verified_file_contents': verified, 'created_local': local_now()}
    write_json(experiment / 'backup_receipt.json', receipt)
    return receipt


def publish_documents(experiment, protocol, receipt):
    destination = Path(protocol['milestone_report_dir'])
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('RESULTS.md', 'comparison.json', 'geometry_verification.json',
                 'backup_receipt.json', 'supervisor_state.json'):
        shutil.copy2(experiment / name, destination / name)
    text = (experiment / 'RESULTS.md').read_text()
    text += f"\n运行与恢复目录：`{experiment}`。\n\n备份：`{receipt['archive']}`；SHA256 `{receipt['sha256']}`，逐文件内容核验{receipt['verified_file_contents']}项通过。\n"
    (destination / 'README.md').write_text(text)
    result = json.loads((experiment / 'comparison.json').read_text())
    counts = '；'.join(f"{arm} {info['summaries']['overall']['success']}/286、大图{info['summaries']['large']['success']}/44"
                       for arm, info in result['arms'].items())
    ledger = Path(protocol['collaboration_ledger'])
    content = ledger.read_text()
    status = f"当前状态：{local_now()}，密采样多跨度全样本A/B完成：{counts}。备份逐文件核验通过，未替换现用模型；详见{destination}/README.md。"
    content = '\n'.join(status if line.startswith('当前状态：') else line for line in content.splitlines())
    content += f"\n\n### {local_now()} — 密采样多跨度A/B训练与验收完成\n\n- {counts}。\n- 各100,440训练/11,160验证；真实新网络各286任务，完整gain/loss及安全记录见 `{destination}`。\n- 备份 `{receipt['archive']}`，SHA256 `{receipt['sha256']}`。无自动延长、模型替换或GitHub上传。\n"
    ledger.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    protocol = json.loads((experiment / 'protocol.json').read_text())
    assert sha(experiment / 'baseline_continuity.json') == protocol['baseline_continuity_sha256']
    assert sha(experiment / 'training_smoke.json') == protocol['training_smoke_sha256']
    for name, expected in protocol['inputs_sha256'].items():
        assert sha(experiment / 'inputs' / name) == expected, name
    for name, expected in json.loads((experiment / 'source_snapshot_manifest.json').read_text())['files_sha256'].items():
        assert sha(ROOT / name) == expected, name
    status_path = experiment / 'supervisor_state.json'
    state = {'status': 'running', 'started_local': local_now(), 'completed_stages': [],
             'protocol_sha256': sha(experiment / 'protocol.json')}
    if status_path.exists():
        old = json.loads(status_path.read_text())
        assert old['protocol_sha256'] == state['protocol_sha256']
        state['completed_stages'] = old['completed_stages']
    write_json(status_path, state)

    def stage(name, command, log, complete):
        if complete():
            if name not in state['completed_stages']:
                state['completed_stages'].append(name)
            write_json(status_path, state)
            return
        state.update(current_stage=name, command=command, updated_local=local_now())
        write_json(status_path, state)
        print(json.dumps({'stage': name, 'event': 'started', 'local': local_now()}), flush=True)
        with log.open('a') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, check=True)
        assert complete(), name
        state['completed_stages'].append(name)
        write_json(status_path, state)

    try:
        for arm in ARMS:
            run = experiment / arm
            run.mkdir(exist_ok=True)
            command = [sys.executable, '-u', str(ROOT / 'training/posttrain_p37_native_hw.py'),
                '--run-dir', str(run), '--data-dir', protocol['data_dir'],
                '--init-checkpoint', str(experiment / 'inputs/epoch15.pth'),
                '--split-path', str(experiment / 'inputs/split.npz'),
                '--baseline-report', str(experiment / 'inputs/source_report.json'),
                '--expected-source-epoch', '15', '--epochs', '1', '--lr', '.00005',
                '--batch-size', '32', '--max-pixels-per-batch', '196608',
                '--checkpoint-every-batches', '1000', '--decoder-mode', 'adaptive_fallback',
                '--progress-smoothing', '7', '--batch-plan-epoch-offset', '15',
                '--route-continuity-diagnostics', '--experiment-name', arm]
            for key, value in protocol['arms'][arm].items():
                command += ['--' + key.replace('_', '-')]
                command += [str(v) for v in value] if isinstance(value, list) else [str(value)]
            saved = run / 'checkpoints/latest_training_state.pth'
            if saved.exists():
                command += ['--resume-state', str(saved)]
            def trained(run=run):
                path = run / 'run_manifest.json'
                return path.exists() and json.loads(path.read_text())['status'] == 'complete'
            stage(f'train_{arm}', command, run / 'training.log', trained)
        for arm in ARMS:
            run = experiment / arm
            command = [sys.executable, '-u', str(ROOT / 'benchmarks/eval_p37_fixed_decoder.py'),
                '--checkpoint', str(run / 'checkpoints/best_p37.pth'),
                '--source-report', str(experiment / 'inputs/source_report.json'),
                '--baseline-report', str(experiment / 'inputs/baseline274_report.json'),
                '--split-path', str(experiment / 'inputs/split.npz'), '--output-dir', str(run / 'evaluation')]
            if (run / 'evaluation/manifest.json').exists():
                command += ['--resume']
            def evaluated(run=run):
                path = run / 'evaluation/report.json'
                return path.exists() and json.loads(path.read_text())['manifest']['status'] == 'complete'
            stage(f'eval_{arm}', command, run / 'evaluation.log', evaluated)
        verify_geometry(experiment)
        compare(experiment)
        state.update(status='backing_up', current_stage='backup', results_finished_local=local_now())
        write_json(status_path, state)
        receipt = backup(experiment)
        state.update(status='complete', current_stage='complete', finished_local=local_now(),
                     comparison_sha256=sha(experiment / 'comparison.json'), backup=receipt)
        write_json(status_path, state)
        publish_documents(experiment, protocol, receipt)
        print(json.dumps({'status': 'complete', 'backup': receipt}), flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error), failed_local=local_now())
        write_json(status_path, state)
        failure = f"失败时间：{state['failed_local']}\n\n阶段：{state.get('current_stage')}\n\n错误：{error!r}\n\n保留断点，修复后可用相同服务命令恢复；无无限自动重启。\n"
        (experiment / 'FAILED.md').write_text(failure)
        destination = Path(protocol['milestone_report_dir'])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'FAILED.md').write_text(failure)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
