"""Content fingerprints for new experiments; no secrets or connection details."""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path
import torch


def file_sha256(path):
    with open(path, 'rb') as f:
        # ``hashlib.file_digest`` was added in Python 3.11.  The A800
        # environment used for the overnight training pipeline still runs
        # Python 3.10, so keep provenance hashing identical across runtimes.
        if hasattr(hashlib, 'file_digest'):
            return hashlib.file_digest(f, 'sha256').hexdigest()
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def write_run_manifest(output_dir, args, source_files, artifacts):
    output = Path(output_dir)
    manifest = output / 'run_manifest.json'
    if manifest.exists():
        raise FileExistsError(f'refusing to overwrite experiment provenance: {manifest}; use a new save-dir')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite a nonempty experiment directory: {output}; use a new save-dir')
    # Resolve/fingerprint inputs BEFORE creating a partial experiment directory.
    # Missing artifacts must not strand an apparently valid source snapshot.
    artifact_records={name: {'path': str(Path(path).resolve()), 'sha256': file_sha256(path)}
                      for name,path in artifacts.items()}
    source_records=[(Path(item).resolve(),Path(item).read_bytes()) for item in source_files]
    snapshot = output / 'source_snapshot'
    snapshot.mkdir(parents=True, exist_ok=True)
    records = {}
    for i, (path,content) in enumerate(source_records):
        dest = snapshot / f'{i:02d}_{path.name}'
        dest.write_bytes(content)
        records[str(path)] = {'sha256': hashlib.sha256(content).hexdigest(),
                              'snapshot': str(dest.relative_to(output))}
    data = {'args': vars(args), 'source_files': records,
            'artifacts': artifact_records,
            'python': platform.python_version(), 'torch': str(torch.__version__),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    return file_sha256(manifest)
