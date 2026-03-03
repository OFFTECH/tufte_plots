"""Backend adapters for TuftePlots."""

from .base import BackendAdapter
from .matplotlib_adapter import MatplotlibAdapter
from .plotly_adapter import PlotlyAdapter
from .seaborn_adapter import SeabornAdapter
from .grid import small_multiples, calculate_grid_dimensions

__all__ = [
    "BackendAdapter",
    "MatplotlibAdapter",
    "PlotlyAdapter",
    "SeabornAdapter",
    "small_multiples",
    "calculate_grid_dimensions",
]
