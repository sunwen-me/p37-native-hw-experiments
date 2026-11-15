# P37 原生 H×W 实验：源码与原始记录

本仓库保存截至 2026-11-16（北京时间）的源码快照、实验配置、训练记录与原始评估数值。文件直接纳入 Git。

```bash
git clone https://github.com/sunwen-me/p37-native-hw-experiments.git
cd p37-native-hw-experiments
```

## 源码

| 目录 | 内容 |
|---|---|
| `inference/` | 本轮实验冻结版本的模型、推理与解码实现 |
| `training/` | 本轮实验冻结版本的数据生成/加载、训练入口、目标函数与监管脚本 |
| `benchmarks/` | 本轮实验冻结版本的评估、数值诊断与核验程序 |
| `tests/` | 本轮实验冻结版本的测试 |
| `requirements.txt` | 原源码快照中的依赖清单 |
| `source_snapshots/previous_increment/` | 上一轮进度增量实验所用源码 |
| `source_snapshots/epoch15/` | epoch15 续训运行所用源码 |

## 配置、训练记录与评估文件

| 目录 | 内容 |
|---|---|
| `runs/multispan/` | 本轮 A_cfm / B_dense_multispan 的 protocol、run_manifest、history、训练与评估日志、逐任务评估 JSON、全部候选路径 NPZ |
| `runs/previous_increment/` | 上一轮 A_cfm / B_route_increment 的对应记录 |
| `runs/epoch15/` | epoch15 与父运行训练记录、评估 JSON 和候选路径 NPZ |
| `inputs/` | 输入评估报告、历史对照报告、固定 latent 与任务 split |
| `dataset/generation_manifest.json` | 数据生成参数、尺寸、任务计数、几何设置与数据文件哈希 |
| `raw_diagnostics/` | 数值诊断程序输出 |
| `checks/` | 既有测试日志、训练 smoke 与几何复核记录 |
| `PROVENANCE.json` | 文件来源、原文件哈希、仓库内哈希与 JSON 字段删除记录 |
| `MANIFEST.json` | 仓库内资料文件的 SHA-256 清单 |

每组评估的逐任务和逐候选记录位于其 `evaluation/report.json`；对应路径数组位于 `evaluation/paths/`。训练历史位于 `history.json`，训练日志位于 `training.log`，运行配置位于 `run_manifest.json`。

## 资料处理规则

源码、训练日志、候选路径 NPZ 与未投影记录逐字保留。根目录 `.gitignore` 为本公开仓库单独配置，允许将实验记录与候选路径纳入 Git。

配置/评估 JSON 中叙述性的 hypothesis、weight_selection、limitations、interpretation、recommendation、conclusion、next_steps 或 analysis 字段已删除；具体删除路径记录在 `PROVENANCE.json`。其余字段、数值、逐任务和逐候选记录保持原样。五组评估的 `tasks` 和 `summaries` 已逐项与原文件核对。

历史分析报告、聊天记录、结论和改进建议未加入本仓库。权重 checkpoint、全量训练数据和稠密预测场缓存未加入；相关哈希保留在配置和运行记录中。文件中的原本机绝对路径属于运行元数据。

## 完整性核验

可在仓库根目录检查全部资料文件：

```bash
python - <<'PYVERIFY'
import hashlib, json
from pathlib import Path
for name, expected in json.loads(Path('MANIFEST.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('All file hashes match.')
PYVERIFY
```
