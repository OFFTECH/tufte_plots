"""
Backend adapters for TuftePlots.

This module contains the abstract BackendAdapter base class and concrete
implementations for matplotlib, plotly, and seaborn backends.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import logging

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    matplotlib = None

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

from tufteplots.theme import TufteTheme
from tufteplots.label_positioner import LabelPositioner
from tufteplots.color_utils import (
    normalize_color,
    ensure_hex_palette,
    hex_to_rgb_normalized,
)

logger = logging.getLogger("tufteplots")

# Type alias for figure types
FigureType = Any  # Union of matplotlib.figure.Figure, plotly.graph_objects.Figure, etc.

class BackendAdapter(ABC):
    """
    Abstract base class for backend adapters.

    This class defines the interface that all backend adapters must implement
    to provide consistent Tufte styling across different plotting libraries.
    """

    def __init__(self, theme: TufteTheme):
        """
        Initialize the adapter with a theme.

        Args:
            theme: The TufteTheme to apply to figures.
        """
        self.theme = theme

    @abstractmethod
    def apply_theme(self, figure: Any, theme: TufteTheme) -> Any:
        """
        Apply Tufte theme to a figure.

        Args:
            figure: The figure object to style.
            theme: The TufteTheme containing styling parameters.

        Returns:
            The styled figure object.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_chartjunk(self, figure: Any) -> Any:
        """
        Remove unnecessary visual elements from the figure.

        This includes removing grid lines, chart borders, and other
        decorative elements that don't convey data information.

        Args:
            figure: The figure object to modify.

        Returns:
            The modified figure object.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_range_frame(self, figure: Any, data_range: Tuple[float, float]) -> Any:
        """
        Set axis limits to match the data range only.

        Args:
            figure: The figure object to modify.
            data_range: Tuple of (min, max) values for the data.

        Returns:
            The modified figure object.
        """
        raise NotImplementedError

    @abstractmethod
    def add_direct_labels(
        self, figure: Any, labels: List[str], positions: List[Tuple[float, float]]
    ) -> Any:
        """
        Add labels directly on the plot.

        Args:
            figure: The figure object to modify.
            labels: List of label strings.
            positions: List of (x, y) coordinates for each label.

        Returns:
            The modified figure object.
        """
        raise NotImplementedError

    @abstractmethod
    def create_line_plot(self, data: Any, x: str, y: str, **kwargs) -> Any:
        """
        Create a styled line plot.

        Args:
            data: The data to plot (typically a pandas DataFrame).
            x: Column name for x-axis data.
            y: Column name for y-axis data.
            **kwargs: Additional plotting parameters.

        Returns:
            The created figure object.
        """
        raise NotImplementedError

    @abstractmethod
    def create_scatter_plot(self, data: Any, x: str, y: str, **kwargs) -> Any:
        """
        Create a styled scatter plot.

        Args:
            data: The data to plot (typically a pandas DataFrame).
            x: Column name for x-axis data.
            y: Column name for y-axis data.
            **kwargs: Additional plotting parameters.

        Returns:
            The created figure object.
        """
        raise NotImplementedError

    @abstractmethod
    def create_bar_plot(self, data: Any, x: str, y: str, **kwargs) -> Any:
        """
        Create a styled bar plot.

        Args:
            data: The data to plot (typically a pandas DataFrame).
            x: Column name for x-axis data.
            y: Column name for y-axis data.
            **kwargs: Additional plotting parameters.

        Returns:
            The created figure object.
        """
        raise NotImplementedError

    @abstractmethod
    def create_histogram(self, data: Any, column: str, **kwargs) -> Any:
        """
        Create a styled histogram.

        Args:
            data: The data to plot (typically a pandas DataFrame).
            column: Column name for the data to histogram.
            **kwargs: Additional plotting parameters.

        Returns:
            The created figure object.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, figure: Any, filepath: str, **kwargs) -> None:
        """
        Export figure to a file.

        Args:
            figure: The figure object to save.
            filepath: Path where the file should be saved.
            **kwargs: Additional save parameters (e.g., dpi, format).
        """
        raise NotImplementedError

    @abstractmethod
    def show(self, figure: Any) -> None:
        """
        Display figure in the current environment.

        Args:
            figure: The figure object to display.
        """
        raise NotImplementedError
