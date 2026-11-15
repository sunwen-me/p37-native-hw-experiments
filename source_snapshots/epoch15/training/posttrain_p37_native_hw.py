"""Bounded full-corpus P37 supervised post-training with resumable batch cursors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / 'training/source/map_flow_matching_p1/scripts'),
               str(PROJECT / 'training/source/map_flow_matching')]
import train_p37_ordered_field as trainer
from utils.dataset import PathPlanningDataset, ShapeBucketBatchSampler


def sha(path):
    return trainer.file_sha256(path)


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def capture_rng(generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'train_generator': generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    generator.set_state(state['train_generator'])


def epoch_batches(dataset, ids, batch_size, max_pixels, seed, epoch, shuffle=True):
    generator = torch.Generator().manual_seed(seed + 700_001 + epoch)
    return list(ShapeBucketBatchSampler(
        dataset, batch_size, indices=ids, shuffle=shuffle, generator=generator,
        max_pixels_per_batch=max_pixels,
    ))


def batch_plan_sha(batches):
    return hashlib.sha256(json.dumps(batches, separators=(',', ':')).encode()).hexdigest()


def validate_resume_configuration(saved, requested):
    # A continuation may increase only its total epoch target and relocate its
    # output/source snapshot. All modeling, data, loss, seed, and optimizer
    # hyperparameters must remain identical.
    if set(saved) != set(requested):
        raise ValueError('resume configuration keys differ')
    relocatable = {'source_project', 'save_dir'}
    for key in saved:
        if key == 'epochs':
            if requested[key] < saved[key]:
                raise ValueError('resume epoch target cannot decrease')
        elif key not in relocatable and saved[key] != requested[key]:
            raise ValueError(f'resume configuration changed: {key}')


def validate_resume_manifest(old, requested):
    immutable = ('source_sha256', 'data_sha256', 'split_sha256',
                 'source_checkpoint_sha256', 'baseline_report_sha256',
                 'learning_rate', 'max_pixels_per_batch', 'batch_size')
    for key in immutable:
        if old[key] != requested[key]:
            raise ValueError(f'resume protocol changed: {key}')
    if requested['posttrain_epochs'] < old['posttrain_epochs']:
        raise ValueError('resume epoch target cannot decrease')


def make_loader(dataset, batches, seed):
    # DataLoader's iterator seed never consumes the model/global RNG cursor.
    return DataLoader(dataset, batch_sampler=batches, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def make_model(config):
    keys = ('map_size', 'map_height', 'map_width', 'horizon', 'latent_dim', 'hidden_dim',
            'decode_beta', 'decode_power', 'decode_threshold', 'distance_scale',
            'robot_radius_grid', 'clearance_temperature', 'clearance_offset',
            'decode_clearance_power', 'decode_continuity_beta', 'accept_threshold',
            'decode_terminal_support_window', 'decode_terminal_support_fraction',
            'decode_max_terminal_jump', 'decode_max_step_grid', 'architecture',
            'transformer_window_size', 'decoder_mode')
    return trainer.OrderedDenseFieldFlow(**{key: config[key] for key in keys if key in config})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--init-checkpoint', type=Path, required=True)
    parser.add_argument('--split-path', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-pixels-per-batch', type=int, default=196608)
    parser.add_argument('--checkpoint-every-batches', type=int, default=1000)
    parser.add_argument('--resume-state', type=Path)
    cli = parser.parse_args()
    if min(cli.epochs, cli.batch_size, cli.max_pixels_per_batch, cli.checkpoint_every_batches) < 1:
        parser.error('epochs, batch sizes and checkpoint interval must be positive')
    run, data = cli.run_dir.resolve(), cli.data_dir.resolve()
    output = run / 'checkpoints'
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = run / 'run_manifest.json'
    if manifest_path.exists() and not cli.resume_state:
        raise ValueError('existing run: use --resume-state or a new directory')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    expected = json.loads((data / 'generation_manifest.json').read_text())['sha256']
    for name, digest in expected.items():
        if sha(data / name) != digest:
            raise ValueError(f'dataset hash mismatch: {name}')
    source_sha = sha(cli.init_checkpoint)
    baseline = json.loads(cli.baseline_report.read_text())
    if source_sha != baseline['manifest']['binding']['checkpoint_sha256']:
        raise ValueError('initial weights do not match the frozen-model baseline')
    if sha(cli.split_path) != baseline['manifest']['binding']['split_sha256']:
        raise ValueError('split differs from the frozen-model baseline')
    source = torch.load(cli.init_checkpoint, map_location='cpu', weights_only=False)
    if source['config'].get('architecture') != 'transformer' or source['epoch'] != 5:
        raise ValueError('expected the audited epoch-5 Transformer checkpoint')
    config = dict(source['config'])
    config.update({
        'source_checkpoint_epoch': 5, 'source_checkpoint_sha256': source_sha,
        'source_checkpoint_hash': source_sha, 'source_checkpoint': str(cli.init_checkpoint.resolve()),
        'source_model_config': source['config'], 'original_model_config': source['config'],
        'source_project': str(PROJECT / 'training/source/map_flow_matching'),
        'data_path': str(data / 'train_data.npz'), 'route_labels': str(data / 'p37_routes.npz'),
        'split_path': str(cli.split_path.resolve()), 'save_dir': str(output),
        'horizon': 512, 'decode_horizon': 512, 'decoder_mode': 'legacy',
        'rasterization_mode': 'segments', 'field_loss_reduction': 'support_balanced',
        'subpixel_rasterization': False, 'lr': cli.lr, 'epochs': cli.epochs,
        'batch_size': cli.batch_size, 'max_pixels_per_batch': cli.max_pixels_per_batch,
        'num_workers': 0, 'progress_every_batches': 50,
        'geometry_val_max_tasks': 0, 'geometry_val_batch_size': 1,
        'geometry_val_max_pixels': 1_048_576, 'geometry_val_candidates': 4,
        'geometry_val_steps': 100, 'geometry_best_metric': 'deployment_success_rate',
        'supervised_posttraining': True, 'local_continuation': False,
        'source_optimizer_state_note': 'New supervision objective; fresh AdamW, no old moments restored.',
        'endpoint_protocol': 'exact_hard_endpoints_v2',
    })
    seed = int(config['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    raw = np.load(data / 'p37_routes.npz', allow_pickle=False)
    labels = {key: torch.from_numpy(raw[key]).float() for key in ('fine_routes', 'valid')}
    config['teacher_horizon'] = labels['fine_routes'].shape[2]
    base = PathPlanningDataset(
        str(data / 'train_data.npz'), horizon=config['teacher_horizon'], normalize=True,
        augment=False, distance_scale=config['distance_scale'],
        validate_geometry=False, cache_distance_fields=True,
    )
    train_ids, val_ids, hashes = trainer.load_map_disjoint_split(base.maps, cli.split_path)
    if (len(train_ids), len(val_ids), len(base)) != (100440, 11160, 111600):
        raise ValueError('post-training requires the full 100440/11160 corpus')
    if len(labels['fine_routes']) != len(base):
        raise ValueError('route labels and dataset length disagree')
    with np.load(data / 'safety_audit.npz', allow_pickle=False) as audit:
        if (not np.array_equal(audit['map_hashes'], hashes) or not audit['safe'].all()
                or not np.all(audit['minimum_clearance_lower_bound'] > base.geometry.radius_grid)):
            raise ValueError('trusted safety audit does not match the corpus')
    cohorts = baseline['manifest']['cohorts']
    fixed_name = next(name for name, ids in cohorts.items() if len(ids) == 26)
    geometry_ids = cohorts[fixed_name]
    if not set(geometry_ids).issubset(set(val_ids.tolist())):
        raise ValueError('geometry tasks must belong to validation')
    coverage = Counter(f'{base.shape_at(i)[0]}x{base.shape_at(i)[1]}' for i in geometry_ids)
    if len(coverage) != 13 or set(coverage.values()) != {2}:
        raise ValueError('expected 26 fixed geometry tasks, two per shape')
    config['robot_radius_grid'] = base.geometry.radius_grid
    config['split_audit'] = {'train_tasks': len(train_ids), 'val_tasks': len(val_ids),
                             'map_overlap': 0, 'geometry_val_tasks': 26}
    files = {str(path.relative_to(PROJECT)): sha(path) for path in PROJECT.rglob('*.py')
             if '__pycache__' not in path.parts}
    manifest = {
        'status': 'materializing', 'created_local': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'training_pid': os.getpid(), 'run_dir': str(run), 'source_root': str(PROJECT),
        'source_sha256': files, 'data_sha256': expected, 'source_checkpoint_sha256': source_sha,
        'source_checkpoint_epoch': 5, 'split_sha256': sha(cli.split_path),
        'baseline_report_sha256': sha(cli.baseline_report), 'baseline_report': str(cli.baseline_report.resolve()),
        'source_validation_loss_old_objective': source.get('loss'),
        'posttrain_epochs': cli.epochs, 'train_tasks': len(train_ids), 'validation_tasks': len(val_ids),
        'geometry_tasks': geometry_ids, 'geometry_per_shape': dict(coverage),
        'objective': 'continuous segments + support_balanced CFM', 'primary_decoder': 'legacy_512',
        'geometry_best_metric': 'deployment_success_rate', 'fresh_optimizer': True,
        'learning_rate': cli.lr, 'batch_size': cli.batch_size, 'max_pixels_per_batch': cli.max_pixels_per_batch,
        'checkpoint_every_batches': cli.checkpoint_every_batches, 'first_checkpoint_batches': 50,
        'network_architecture_changed': False, 'train_all_network_parameters': True,
        'rl': False, 'holdout_used': False, 'last_completed_epoch': 0,
        'optimizer_steps': 0, 'gpu': torch.cuda.get_device_name(device), 'torch': str(torch.__version__),
        'cuda': torch.version.cuda, 'tf32': True, 'cudnn_benchmark': False,
        'old_loss_comparable': False,
    }
    resumed = None
    if cli.resume_state:
        old = json.loads(manifest_path.read_text())
        validate_resume_manifest(old, manifest)
        manifest = old
        resumed = torch.load(cli.resume_state, map_location='cpu', weights_only=False)
        validate_resume_configuration(resumed['config'], config)
    atomic_json(manifest, manifest_path)
    dataset = trainer.MaterializedDataset(base)
    model = make_model(config).to(device)
    model.load_state_dict((resumed or source)['model_state_dict'], strict=True)
    model.vector_field.latent_scale = float(config.get('latent_scale', 1.))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cli.lr, weight_decay=1e-4)
    train_generator = torch.Generator(device=device).manual_seed(seed + 900_017)
    source = None
    best_val = math.inf
    best_geometry = baseline['cohorts'][fixed_name]['summaries']['legacy_512']['deployment_success_rate']
    cursor = {'epoch': 1, 'stage': 'train', 'completed_batches': 0,
              'train_total': 0., 'train_samples': 0, 'optimizer_steps': 0}
    if resumed:
        optimizer.load_state_dict(resumed['optimizer_state_dict'])
        restore_rng(resumed['rng_state'], train_generator)
        cursor = resumed['cursor']
        best_val = resumed['best_val_loss']
        best_geometry = resumed['best_geometry_deployment_success']
        resumed = None
    else:
        initial = {'epoch': 0, 'variant': model.variant, 'model_state_dict': model.state_dict(),
                   'config': config, 'loss': None, 'posttraining_optimizer_steps': 0,
                   'geometry_best_metric': 'deployment_success_rate',
                   'geometry_val': {'deployment_success_rate': best_geometry,
                                    'reference': str(cli.baseline_report.resolve())}}
        atomic_torch_save(initial, output / 'initial_model.pth')
        atomic_torch_save(initial, output / 'best_p37_geometry.pth')

    def save_state():
        snapshot = {'epoch': cursor['epoch'], 'variant': model.variant,
                    'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'config': config, 'rng_state': capture_rng(train_generator), 'cursor': dict(cursor),
                    'best_val_loss': best_val, 'best_geometry_deployment_success': best_geometry,
                    'loss': manifest.get('latest_val_loss'), 'geometry_best_metric': 'deployment_success_rate'}
        target = output / 'latest_training_state.pth'
        atomic_torch_save(snapshot, target)
        manifest.update({'latest_checkpoint_sha256': sha(target), 'cursor': dict(cursor),
                         'optimizer_steps': cursor['optimizer_steps'],
                         'updated_local': time.strftime('%Y-%m-%d %H:%M:%S %Z')})
        atomic_json(manifest, manifest_path)

    args = SimpleNamespace(**config)
    validation_plan = epoch_batches(dataset, val_ids.tolist(), cli.batch_size,
                                    cli.max_pixels_per_batch, seed, 0, shuffle=False)
    history_path = run / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    save_state()
    start_epoch = cursor['epoch'] + (cursor['stage'] == 'epoch_complete')
    for epoch in range(start_epoch, cli.epochs + 1):
        started = time.perf_counter()
        plan = epoch_batches(dataset, train_ids.tolist(), cli.batch_size,
                             cli.max_pixels_per_batch, seed, epoch)
        flat_ids = np.asarray([task for batch in plan for task in batch])
        if not np.array_equal(np.sort(flat_ids), np.sort(train_ids)):
            raise RuntimeError('batch plan dropped or duplicated training tasks')
        if cursor['epoch'] != epoch or cursor['stage'] == 'epoch_complete':
            cursor.update(epoch=epoch, stage='train', completed_batches=0, train_total=0., train_samples=0)
        digest = batch_plan_sha(plan)
        if cursor.get('batch_plan_sha256') and cursor['completed_batches'] and cursor['batch_plan_sha256'] != digest:
            raise ValueError('resume batch order differs from the saved cursor')
        cursor['batch_plan_sha256'] = digest
        manifest.update(status='running', training_pid=os.getpid(), current_epoch=epoch,
                        train_batches_per_epoch=len(plan), validation_batches_per_epoch=len(validation_plan))
        atomic_json(manifest, manifest_path)
        print(json.dumps({'stage': 'epoch_start', 'epoch': epoch, 'train_batches': len(plan),
                          'train_tasks': 100440, 'validation_tasks': 11160,
                          'resume_completed_batches': cursor['completed_batches'],
                          'lr': cli.lr, 'new_objective': config['field_loss_reduction']}), flush=True)
        while cursor['completed_batches'] < len(plan):
            begin = cursor['completed_batches']
            interval = 50 if begin == 0 else cli.checkpoint_every_batches
            end = min(len(plan), begin + interval)
            chunk = plan[begin:end]
            samples = sum(map(len, chunk))
            loss = trainer.run_epoch(
                model, make_loader(dataset, chunk, seed + epoch), labels, device, args, optimizer,
                generator=train_generator,
            )
            if not math.isfinite(loss):
                raise FloatingPointError('non-finite training loss')
            cursor.update(completed_batches=end, train_total=cursor['train_total'] + loss * samples,
                          train_samples=cursor['train_samples'] + samples,
                          optimizer_steps=cursor['optimizer_steps'] + len(chunk),
                          stage='validate' if end == len(plan) else 'train')
            manifest.update(latest_train_mean_loss=cursor['train_total'] / cursor['train_samples'],
                            peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024 ** 3)
            save_state()
            print(json.dumps({'stage': 'training_checkpoint', 'epoch': epoch,
                              'batch': end, 'batches': len(plan), 'samples': cursor['train_samples'],
                              'mean_loss': manifest['latest_train_mean_loss'],
                              'optimizer_steps': cursor['optimizer_steps'],
                              'checkpoint': str(output / 'latest_training_state.pth')}), flush=True)
        if cursor['train_samples'] != 100440:
            raise RuntimeError('epoch did not consume all training tasks')
        with torch.no_grad():
            val_loss = trainer.run_epoch(
                model, make_loader(dataset, validation_plan, seed), labels, device, args,
                generator=torch.Generator(device=device).manual_seed(seed + 300_003),
            )
        if not math.isfinite(val_loss):
            raise FloatingPointError('non-finite validation loss')
        manifest['latest_val_loss'] = val_loss
        weights = {'epoch': epoch, 'variant': model.variant, 'model_state_dict': model.state_dict(),
                   'config': config, 'loss': val_loss, 'posttraining_optimizer_steps': cursor['optimizer_steps']}
        if val_loss < best_val:
            best_val = val_loss
            atomic_torch_save(weights, output / 'best_p37.pth')
        geometry = trainer.geometry_success(model, base, geometry_ids, device, args)
        score = geometry['deployment_success_rate']
        if score > best_geometry:
            best_geometry = score
            atomic_torch_save(weights | {'geometry_val': geometry,
                                        'geometry_best_metric': 'deployment_success_rate'},
                              output / 'best_p37_geometry.pth')
        row = {'epoch': epoch, 'train_loss': cursor['train_total'] / cursor['train_samples'],
               'val_loss': val_loss, 'train_tasks': 100440, 'validation_tasks': 11160,
               'geometry': geometry, 'epoch_seconds_this_invocation': time.perf_counter() - started,
               'optimizer_steps': cursor['optimizer_steps']}
        history = [item for item in history if item['epoch'] != epoch] + [row]
        atomic_json(history, history_path)
        cursor['stage'] = 'epoch_complete'
        manifest.update(last_completed_epoch=epoch, best_val_loss=best_val,
                        best_geometry_deployment_success=best_geometry)
        save_state()
        print(json.dumps({'stage': 'epoch_complete', **row}), flush=True)
    manifest.update(status='complete', finished_local=time.strftime('%Y-%m-%d %H:%M:%S %Z'))
    atomic_json(manifest, manifest_path)


if __name__ == '__main__':
    main()
