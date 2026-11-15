import torch

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
    assert tuple(loaded.state_dict()) == tuple(original.state_dict())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


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
