"""Add new diagnostics to old checkpoint-bound caches, with zero network calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'training'), str(ROOT / 'training/source/map_flow_matching_p1'),
               str(ROOT / 'training/source/map_flow_matching')]
from run_p37_increment_ab import sha, write_json, local_now
from utils.dataset import PathPlanningDataset
from route_field_diagnostics import route_field_diagnostics
from safe_field_decoder import support_weighted_progress_smoothing
from p37_model import endpoint_fields
from posttrain_p37_native_hw import make_model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    protocol = json.loads((experiment / 'protocol.json').read_text())
    out = experiment / 'baseline_continuity.json'
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    source = json.loads((experiment / 'inputs/source_report.json').read_text())
    base = PathPlanningDataset(source['manifest']['binding']['data_path'], normalize=True,
        augment=False, distance_scale=8., validate_geometry=False, cache_distance_fields=True)
    initial = torch.load(experiment / 'inputs/epoch15.pth', map_location='cpu', weights_only=False)
    model = make_model(initial['config']).cuda().eval().requires_grad_(False)
    model.load_state_dict(initial['model_state_dict'], strict=True)
    calls = [0]
    model.vector_field.register_forward_pre_hook(lambda *args: calls.__setitem__(0, calls[0] + 1))
    groups = {}
    for name, binding in protocol['reference_caches'].items():
        report_path = experiment / 'inputs' / binding['report']
        report = json.loads(report_path.read_text())
        checkpoint_sha = report['manifest']['binding']['checkpoint_sha256']
        if 'checkpoint' in binding:
            assert sha(Path(binding['checkpoint'])) == checkpoint_sha
        rows = []
        for task in report['tasks']:
            tid = task['task_id']
            path = Path(binding['fields']) / f'task_{tid:06d}.npz'
            with np.load(path) as cached:
                raw = cached['fields']
                field_hash = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
                assert field_hash == task['fields_sha256']
                if 'checkpoint_sha256' in cached:
                    assert str(cached['checkpoint_sha256']) == checkpoint_sha
            fields = torch.from_numpy(raw).cuda()
            if fields.ndim == 4:
                fields = fields[None]
            batch, candidates, _, height, width = fields.shape
            assert batch == 1 and candidates == 4
            item = base[tid]
            start, goal = (item[key][None].cuda() for key in ('start', 'goal'))
            flat = fields.flatten(0, 1)
            corrected = fields.clone()
            corrected[:, :, 1:2] = support_weighted_progress_smoothing(flat[:, :1], flat[:, 1:2], 7).reshape(1, 4, 1, height, width)
            fixed = endpoint_fields(start, goal, height, width)[:, None]
            corrected = corrected * (1 - fixed[:, :, :1]) + fixed
            if not rows:
                maps = torch.cat([item['map'], item['distance_field']], 0)[None].cuda()
                actual = model.decode_fields(maps, start, goal, fields, mode='adaptive_fallback', horizon=512, progress_smoothing=7)
                assert torch.equal(actual['decoder_fields'], corrected)
            rows.append({'task_id': tid, 'shape': task['shape'], 'fields_sha256': field_hash,
                'reference_continuity_diagnostics': route_field_diagnostics(base.paths[tid], corrected[0].cpu().numpy())})
        groups[name] = {'source_report_sha256': sha(report_path), 'checkpoint_sha256': checkpoint_sha,
                        'correction_checked_bitwise_against_decoder': True, 'tasks': rows}
        print(json.dumps({'baseline': name, 'tasks': len(rows), 'network_calls': calls[0]}), flush=True)
    assert calls[0] == 0
    write_json(out, {'status': 'complete', 'finished_local': local_now(), 'network_calls': 0,
        'old_model_diagnostics_only': True, 'new_model_predictions': False,
        'teacher_fields_used_for_decoding': False, 'reference_mode_mismatch_possible': True,
        'groups': groups})


if __name__ == '__main__':
    main()
