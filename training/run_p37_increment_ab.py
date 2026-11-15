"""Run exactly two full one-epoch experiments and fixed-decoder evaluations.

Execute from a frozen source snapshot. Stage manifests/checkpoints support
restarting this supervisor after interruption without mixing model caches.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def local_now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def compare(experiment):
    reports = {arm: json.loads((experiment / arm / "evaluation/report.json").read_text())
               for arm in ("A_cfm", "B_route_increment")}
    histories = {arm: json.loads((experiment / arm / "history.json").read_text()) for arm in reports}
    for arm, history in histories.items():
        assert len(history) == 1 and history[0]["train_tasks"] == 100440 and history[0]["validation_tasks"] == 11160
        assert len(reports[arm]["tasks"]) == 286
    manifests = {arm: json.loads((experiment / arm / "run_manifest.json").read_text()) for arm in reports}
    for key in ("source_checkpoint_sha256", "split_sha256", "learning_rate", "batch_size",
                "max_pixels_per_batch", "batch_plan_epoch_offset", "source_sha256", "data_sha256"):
        assert manifests["A_cfm"][key] == manifests["B_route_increment"][key], key
    assert manifests["A_cfm"]["cursor"]["batch_plan_sha256"] == manifests["B_route_increment"]["cursor"]["batch_plan_sha256"]
    a, b = ({r["task_id"]: r for r in reports[arm]["tasks"]} for arm in reports)
    assert set(a) == set(b)
    gains = [tid for tid in a if b[tid]["deployment_success"] and not a[tid]["deployment_success"]]
    losses = [tid for tid in a if a[tid]["deployment_success"] and not b[tid]["deployment_success"]]
    summary = {"status": "complete", "finished_local": local_now(),
               "arms": {arm: {"train_tasks": histories[arm][0]["train_tasks"],
                               "validation_tasks": histories[arm][0]["validation_tasks"],
                               "train_loss": histories[arm][0]["train_loss"],
                               "validation_loss_own_objective": histories[arm][0]["val_loss"],
                               "summaries": reports[arm]["summaries"],
                               "checkpoint_sha256": sha(experiment / arm / "checkpoints/best_p37.pth")}
                        for arm in reports},
               "B_vs_A_gains": gains, "B_vs_A_losses": losses,
               "equal_batch_order_verified": True, "same_fresh_optimizer_protocol": True,
               "validation_losses_directly_comparable": False,
               "current_baseline_replaced": False, "holdout_used": False,
               "limitations": ["One epoch and one seed are screening evidence, not stable improvement.",
                               "All 286 tasks are reused development cases.",
                               "The auxiliary uses a CFM endpoint estimate; actual integrated samples determine geometry."]}
    write_json(experiment / "comparison.json", summary)
    lines = ["# 路线进度约束：单轮等预算 A/B", "", f"完成：{summary['finished_local']}", "",
             "两组均从 epoch15 初始化、使用新 AdamW、相同随机种子与批次顺序，完整100,440训练/11,160验证。",
             "A为原CFM；B仅增加0.1×沿配对路线弧长采样的进度增量Smooth-L1。",
             "固定adaptive_fallback/smooth7/512，K4/100步，真实重新采样各286条。", "",
             "| 组别 | 全体 | 最大两尺寸 | 相对274新增 | 相对274丢失 | 接受但不安全候选 |",
             "|---|---:|---:|---|---|---:|"]
    for arm, report in reports.items():
        whole, large = (report["summaries"][key] for key in ("overall", "large"))
        lines.append(f"| {arm} | {whole['success']}/286 | {large['success']}/44 | {whole['gains']} | {whole['losses']} | {whole['accepted_unsafe_candidates']} |")
    lines += ["", f"B相对A新增：{gains}；丢失：{losses}。", "",
              "B的验证loss包含新增项，不能和A的总loss直接排名。是否有收益以几何成功、回退任务和机制诊断为准。",
              "本次不自动替换274基线、不延长训练、不使用保留未见尺寸集合，也不上传GitHub。",
              "单轮单种子和复用开发集不能证明稳定提升或真实任意地图质量。详细数据见comparison.json及两组evaluation/report.json。"]
    (experiment / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, required=True)
    args = p.parse_args()
    experiment = args.experiment_dir.resolve()
    protocol = json.loads((experiment / "protocol.json").read_text())
    for name, expected in protocol["inputs_sha256"].items():
        assert sha(experiment / "inputs" / name) == expected, name
    snapshot = json.loads((experiment / "source_snapshot_manifest.json").read_text())
    for name, expected in snapshot["files_sha256"].items():
        assert sha(ROOT / name) == expected, name
    status_path = experiment / "supervisor_state.json"
    state = {"status": "running", "started_local": local_now(), "completed_stages": [],
             "protocol_sha256": sha(experiment / "protocol.json")}
    if status_path.exists():
        old = json.loads(status_path.read_text())
        assert old["protocol_sha256"] == state["protocol_sha256"]
        state["completed_stages"] = old["completed_stages"]
    write_json(status_path, state)

    def run_stage(name, command, log, complete):
        if complete():
            if name not in state["completed_stages"]:
                state["completed_stages"].append(name)
            return
        state.update(status="running", current_stage=name, command=command, updated_local=local_now())
        write_json(status_path, state)
        print(json.dumps({"stage": name, "event": "started", "local": local_now()}), flush=True)
        with log.open("a") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        assert complete(), name
        state["completed_stages"].append(name)
        write_json(status_path, state)
        print(json.dumps({"stage": name, "event": "complete", "local": local_now()}), flush=True)

    try:
        for arm, weight in (("A_cfm", "0"), ("B_route_increment", "0.1")):
            run = experiment / arm
            run.mkdir(exist_ok=True)
            command = [sys.executable, "-u", str(ROOT / "training/posttrain_p37_native_hw.py"),
                "--run-dir", str(run), "--data-dir", protocol["data_dir"],
                "--init-checkpoint", str(experiment / "inputs/epoch15.pth"),
                "--split-path", str(experiment / "inputs/split.npz"),
                "--baseline-report", str(experiment / "inputs/source_report.json"),
                "--expected-source-epoch", "15", "--epochs", "1", "--lr", "0.00005",
                "--batch-size", "32", "--max-pixels-per-batch", "196608", "--checkpoint-every-batches", "1000",
                "--decoder-mode", "adaptive_fallback", "--progress-smoothing", "7",
                "--route-increment-weight", weight, "--route-increment-samples", "128",
                "--route-increment-scale", "32", "--batch-plan-epoch-offset", "15", "--experiment-name", arm]
            saved = run / "checkpoints/latest_training_state.pth"
            if saved.exists():
                command += ["--resume-state", str(saved)]
            def trained(run=run):
                path = run / "run_manifest.json"
                return path.exists() and json.loads(path.read_text()).get("status") == "complete"
            run_stage(f"train_{arm}", command, run / "training.log", trained)
        for arm in ("A_cfm", "B_route_increment"):
            run = experiment / arm
            command = [sys.executable, "-u", str(ROOT / "benchmarks/eval_p37_fixed_decoder.py"),
                       "--checkpoint", str(run / "checkpoints/best_p37.pth"),
                       "--source-report", str(experiment / "inputs/source_report.json"),
                       "--baseline-report", str(experiment / "inputs/baseline274_report.json"),
                       "--split-path", str(experiment / "inputs/split.npz"), "--output-dir", str(run / "evaluation")]
            if (run / "evaluation/manifest.json").exists():
                command.append("--resume")
            def evaluated(run=run):
                path = run / "evaluation/report.json"
                return path.exists() and json.loads(path.read_text())["manifest"]["status"] == "complete"
            run_stage(f"eval_{arm}", command, run / "evaluation.log", evaluated)
        comparison = compare(experiment)
        state.update(status="complete", current_stage="complete", finished_local=local_now(),
                     comparison_sha256=sha(experiment / "comparison.json"))
        write_json(status_path, state)
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        archive = experiment.parent.parent / "backups" / f"p37_route_increment_ab_{stamp}.tar.gz"
        archive.parent.mkdir(exist_ok=True)
        temporary = archive.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz", compresslevel=3) as tar:
            tar.add(experiment, arcname=experiment.name, filter=lambda info: None if "__pycache__" in Path(info.name).parts else info)
        temporary.replace(archive)
        with tarfile.open(archive, "r:gz") as tar:
            names = set(tar.getnames())
            for required in ("comparison.json", "RESULTS.md", "A_cfm/checkpoints/best_p37.pth",
                             "B_route_increment/checkpoints/best_p37.pth"):
                assert f"{experiment.name}/{required}" in names
        write_json(experiment / "backup_receipt.json", {"archive": str(archive), "sha256": sha(archive),
                   "bytes": archive.stat().st_size, "verified_required_members": True, "created_local": local_now()})
        print(json.dumps({"status": "complete", "comparison": str(experiment / "comparison.json"),
                          "backup": str(archive)}), flush=True)
    except BaseException as error:
        state.update(status="failed", failed_local=local_now(), error=repr(error))
        write_json(status_path, state)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
