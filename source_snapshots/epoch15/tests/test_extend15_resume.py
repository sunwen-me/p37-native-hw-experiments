import pytest

from posttrain_p37_native_hw import (
    validate_resume_configuration, validate_resume_manifest,
)


def test_resume_configuration_allows_only_monotonic_epochs_and_output_relocation():
    saved = {"epochs": 5, "lr": 5e-5, "seed": 37, "save_dir": "/old/out",
             "source_project": "/old/source", "loss": "support_balanced"}
    requested = {**saved, "epochs": 15, "save_dir": "/new/out",
                 "source_project": "/new/source"}
    validate_resume_configuration(saved, requested)
    with pytest.raises(ValueError, match="cannot decrease"):
        validate_resume_configuration(saved, {**requested, "epochs": 4})
    with pytest.raises(ValueError, match="lr"):
        validate_resume_configuration(saved, {**requested, "lr": 1e-4})


def test_resume_manifest_keeps_protocol_fixed_while_allowing_target_extension():
    old = {"source_sha256": {"a.py": "one"}, "data_sha256": {"d": "two"},
           "split_sha256": "three", "source_checkpoint_sha256": "four",
           "baseline_report_sha256": "five", "learning_rate": 5e-5,
           "max_pixels_per_batch": 196608, "batch_size": 32, "posttrain_epochs": 5}
    requested = {**old, "posttrain_epochs": 15}
    validate_resume_manifest(old, requested)
    with pytest.raises(ValueError, match="cannot decrease"):
        validate_resume_manifest(old, {**requested, "posttrain_epochs": 4})
    with pytest.raises(ValueError, match="data_sha256"):
        validate_resume_manifest(old, {**requested, "data_sha256": {"d": "changed"}})
