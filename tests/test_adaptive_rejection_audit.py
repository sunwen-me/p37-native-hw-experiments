import importlib.util
from pathlib import Path
import sys

import torch

from adaptive_field_decoder import adaptive_ordered_field_to_path
from test_adaptive_decoder import tail_fields

spec = importlib.util.spec_from_file_location(
    "rejection_audit", Path(__file__).resolve().parents[1] / "benchmarks/p37_adaptive_rejection_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_observation_preserves_outputs_and_separates_overlapping_gates():
    fields = tail_fields(gap=True)
    baseline = adaptive_ordered_field_to_path(*fields, 128)
    tracer = audit.RejectionTrace()
    previous = sys.gettrace()
    traced = tracer.run(adaptive_ordered_field_to_path, *fields, 128)
    assert sys.gettrace() is previous
    for actual, expected in zip(traced[:2], baseline[:2]):
        assert torch.equal(actual, expected)
    for key in baseline[2]:
        assert torch.equal(traced[2][key], baseline[2][key])
    assert len(tracer.events) == int((baseline[2]["adaptive_failure_code"] == 1).sum()) == 1
    proposals = tracer.events[0]["proposals"]
    assert len(proposals) == 7
    assert all(p["failed_checks"] for p in proposals)
    for proposal in proposals:
        expected = []
        if proposal["joint_mass"] <= 1e-5:
            expected.append("mass")
        if proposal["movement_px"] > 4:
            expected.append("movement")
        if proposal["segment_occupancy_min"] <= .1:
            expected.append("segment_occupancy")
        if proposal["segment_clearance_min"] < .5:
            expected.append("segment_clearance")
        assert proposal["failed_checks"] == expected


def test_tracer_restores_callers_state_on_exception():
    tracer = audit.RejectionTrace()
    previous = sys.gettrace()
    try:
        tracer.run(adaptive_ordered_field_to_path, *tail_fields(), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid horizon accepted")
    assert sys.gettrace() is previous
