from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (
    ROOT / "training/source/map_flow_matching",
    ROOT / "training/source/map_flow_matching_p1",
    ROOT / "inference",
):
    sys.path.insert(0, str(directory))
