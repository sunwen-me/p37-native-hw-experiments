"""Package the validated P37 progress decoder configuration with unchanged weights."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from flow_prior import load_flow_prior


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def package_decoder(checkpoint, output, *, profile="smooth7"):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("choose a new output path; existing artifacts are preserved")
    if profile not in {"smooth7", "adaptive_fallback", "bounded_fallback"}:
        raise ValueError("unknown decoder profile")
    _, state = load_flow_prior(checkpoint, "cpu")
    config = dict(state.get("config", {}))
    changes = {"horizon": 512, "decode_horizon": 512,
               "decoder_mode": "scale_aware" if profile == "smooth7" else profile,
               "decode_progress_smoothing": 7}
    metadata = {
        "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_checkpoint": str(checkpoint), "source_sha256": digest(checkpoint),
        "revision": "p37_progress_smooth7_v1" if profile == "smooth7" else f"p37_{profile}_v1",
        "config_changes": changes,
        "original_decode_config": {key: config.get(key) for key in changes},
        "training": False, "weights_changed": False, "safety_thresholds_changed": False,
        "source_files": {name: digest(Path(__file__).parent / name) for name in
                         ("p37_model.py", "safe_field_decoder.py", "adaptive_field_decoder.py", "flow_prior.py")},
    }
    packaged = dict(state)
    packaged["config"] = config | changes
    packaged["decoder_revision"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.pth")
    try:
        torch.save(packaged, temporary)
        model, loaded = load_flow_prior(temporary, "cpu")
        assert model.decode_progress_smoothing == 7 and model.decoder_mode == changes["decoder_mode"]
        assert model.horizon == 512
        assert set(loaded["model_state_dict"]) == set(state["model_state_dict"])
        for key, value in state["model_state_dict"].items():
            assert torch.equal(value, loaded["model_state_dict"][key]), key
        assert digest(checkpoint) == metadata["source_sha256"]
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = metadata | {"output": str(output), "output_sha256": digest(output),
                          "identical_parameter_tensors": len(state["model_state_dict"])}
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smooth7", "adaptive_fallback", "bounded_fallback"), default="smooth7")
    args = parser.parse_args()
    print(json.dumps(package_decoder(args.checkpoint, args.output, profile=args.profile), ensure_ascii=False))


if __name__ == "__main__":
    main()
