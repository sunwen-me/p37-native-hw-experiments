import importlib.util
from pathlib import Path
import sys

import torch

from safe_field_decoder import safe_ordered_field_to_path
from test_adaptive_decoder import tail_fields

spec = importlib.util.spec_from_file_location(
    "post_ab_audit", Path(__file__).resolve().parents[1] / "benchmarks/p37_post_ab_regression_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_first_loss_observer_preserves_production_path_and_diagnostics():
    fields = tail_fields()
    # A wide interior gap loses joint support before terminal bookkeeping.
    fields[0][:, :, :, 100:180] = 0
    expected = safe_ordered_field_to_path(*fields, 128, mode="scale_aware", return_diagnostics=True)
    observer = audit.FirstSupportLossTrace()
    previous = sys.gettrace()
    try:
        sys.settrace(observer.trace)
        actual = safe_ordered_field_to_path(*fields, 128, mode="scale_aware", return_diagnostics=True)
    finally:
        sys.settrace(previous)
    for value, target in zip(actual[:2], expected[:2]):
        assert torch.equal(value, target)
    for key, value in actual[2].items():
        assert torch.equal(value, expected[2][key])
    assert len(observer.events) == 1
    event = observer.events[0]
    assert event["candidate"] == 0
    assert event["step"] == int(actual[2]["first_missing_support_step"][0])
    assert event["local_joint_mass"] <= 1e-5
