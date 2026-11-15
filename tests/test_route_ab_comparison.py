import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "route_ab", Path(__file__).resolve().parents[1] / "training/run_p37_increment_ab.py")
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def fixture_run(root):
    for arm in ("A_cfm", "B_route_increment"):
        run = root / arm
        (run / "evaluation").mkdir(parents=True)
        (run / "checkpoints").mkdir()
        (run / "checkpoints/best_p37.pth").write_bytes(arm.encode())
        ab.write_json(run / "history.json", [{"train_tasks": 100440, "validation_tasks": 11160,
                                             "train_loss": .02, "val_loss": .03}])
        ab.write_json(run / "run_manifest.json", dict.fromkeys(("source_checkpoint_sha256", "split_sha256",
            "learning_rate", "batch_size", "max_pixels_per_batch", "batch_plan_epoch_offset",
            "source_sha256", "data_sha256"), "same") | {"cursor": {"batch_plan_sha256": "order"}})
        rows = [{"task_id": i, "shape": "768x512", "baseline_success": i < 274,
                 "deployment_success": (i < 274 if arm == "A_cfm" else i < 273 or i == 275),
                 "accepted_unsafe_candidates": 0, "selected_accepted_unsafe": False} for i in range(286)]
        ab.write_json(run / "evaluation/report.json", {"tasks": rows,
            "summaries": {"overall": {"success": 274, "gains": [275] if arm != "A_cfm" else [],
                "losses": [273] if arm != "A_cfm" else [], "accepted_unsafe_candidates": 0},
                "large": {"success": 274}}})


def test_ab_discloses_both_gains_and_losses_and_does_not_compare_different_losses(tmp_path):
    fixture_run(tmp_path)
    result = ab.compare(tmp_path)
    assert result["B_vs_A_gains"] == [275]
    assert result["B_vs_A_losses"] == [273]
    assert not result["validation_losses_directly_comparable"]
    assert not result["current_baseline_replaced"]


def test_comparison_rejects_mismatched_budget_or_batch_order(tmp_path):
    fixture_run(tmp_path)
    path = tmp_path / "B_route_increment/run_manifest.json"
    value = json.loads(path.read_text())
    value["cursor"]["batch_plan_sha256"] = "different"
    ab.write_json(path, value)
    with pytest.raises(AssertionError):
        ab.compare(tmp_path)
