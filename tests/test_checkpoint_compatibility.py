import torch
import pytest

from flow_prior import load_flow_prior
from p37_model import OrderedDenseFieldFlow


def test_legacy_square_checkpoint_strict_loads_without_parameter_changes(tmp_path):
    torch.manual_seed(11)
    original = OrderedDenseFieldFlow(map_size=64, hidden_dim=8, latent_dim=4, horizon=8)
    checkpoint = tmp_path / "legacy.pth"
    torch.save({
        "model_state_dict": original.state_dict(),
        "config": {"map_size": 64, "hidden_dim": 8, "latent_dim": 4, "horizon": 8},
    }, checkpoint)
    loaded, _ = load_flow_prior(checkpoint, "cpu")
    assert loaded.decoder_mode == "legacy"
    assert loaded.decode_progress_smoothing == 0
    assert tuple(loaded.state_dict()) == tuple(original.state_dict())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_packaged_progress_repair_preserves_weights_and_enables_planner(tmp_path):
    from package_p37_decoder import package_decoder
    from p37_planner import P37FlowPlanner

    original = OrderedDenseFieldFlow(hidden_dim=8, latent_dim=4, horizon=8)
    source, output = tmp_path / "original.pth", tmp_path / "repair.pth"
    torch.save({"model_state_dict": original.state_dict(), "epoch": 15, "config": {
        "hidden_dim": 8, "latent_dim": 4, "horizon": 8,
    }}, source)
    receipt = package_decoder(source, output)
    planner = P37FlowPlanner(output)
    assert planner.model.decoder_mode == "scale_aware"
    assert planner.model.decode_progress_smoothing == 7
    assert planner.model.horizon == 512
    assert not receipt["training"] and not receipt["weights_changed"]
    for key, value in original.state_dict().items():
        assert torch.equal(planner.model.state_dict()[key], value)
    with pytest.raises(FileExistsError):
        package_decoder(source, output)


def test_new_decoder_mode_survives_checkpoint_load_without_weight_changes(tmp_path):
    original = OrderedDenseFieldFlow(
        hidden_dim=8, latent_dim=4, horizon=8, decoder_mode="scale_aware",
    )
    checkpoint = tmp_path / "v2.pth"
    torch.save({"model_state_dict": original.state_dict(), "config": {
        "hidden_dim": 8, "latent_dim": 4, "horizon": 8, "decoder_mode": "scale_aware",
    }}, checkpoint)
    loaded, _ = load_flow_prior(checkpoint, "cpu")
    assert loaded.decoder_mode == "scale_aware"
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_adaptive_package_strict_loads_with_identical_weights(tmp_path):
    from package_p37_decoder import package_decoder
    original = OrderedDenseFieldFlow(hidden_dim=8, horizon=128)
    source, output = tmp_path / "original.pth", tmp_path / "adaptive.pth"
    torch.save({"model_state_dict": original.state_dict(), "config": {
        "hidden_dim": 8, "horizon": 128}}, source)
    receipt = package_decoder(source, output, profile="adaptive_fallback")
    loaded, _ = load_flow_prior(output, "cpu")
    assert loaded.decoder_mode == "adaptive_fallback" and loaded.horizon == 512
    assert receipt["identical_parameter_tensors"] == len(original.state_dict())
    assert not receipt["weights_changed"] and not receipt["safety_thresholds_changed"]
    for key, value in original.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)
