"""Utilities included in the compact final training bundle.

Keep package imports limited to files shipped alongside this module. The full
research repository also has data generation and visualization utilities, but
training consumes prebuilt datasets and does not need those optional modules.
"""
from .dataset import PathPlanningDataset, collate_fn

__all__ = ["PathPlanningDataset", "collate_fn"]
