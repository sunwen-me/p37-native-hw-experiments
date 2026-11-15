"""Final SOTA deployment wrapper for the Chapter 3 planner."""
from __future__ import annotations
from pathlib import Path
from refinement_planner import FlowRefinementPlanner

HERE = Path(__file__).resolve().parent
DEFAULT_GENERATOR = HERE / "flow_prior.pth"
DEFAULT_REFINER = HERE / "refiner_warm.pth"


def load_planner(device: str = "cuda", *, compile_mode=None):
    """Load the locked endpoint-P37 + warm-refiner deployment."""
    return FlowRefinementPlanner(
        DEFAULT_GENERATOR,
        DEFAULT_REFINER,
        device=device,
        include_base=True,
        compile_mode=compile_mode,
    )


def plan(occupancy, starts_px, goals_px, *, candidates=4, seed=20260921,
         flow_steps=100, device="cuda"):
    """Generate the selected path and all base/refined options.

    Returned safety values are learned estimates, not exact collision
    certificates. Exact geometry is used only by offline evaluation.
    """
    planner = load_planner(device=device)
    return planner.plan(
        occupancy, starts_px, goals_px, candidates=candidates,
        seed=seed, flow_steps=flow_steps,
    )
