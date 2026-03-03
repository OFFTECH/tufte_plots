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

from .base import BackendAdapter, FigureType

class SeabornAdapter(BackendAdapter):
    """
    Adapter for seaborn figures.

    This adapter implements Tufte styling for seaborn figures, handling
    seaborn context, despine, and style settings while preserving seaborn-specific features.
    """

    def __init__(self, theme: TufteTheme):
        """
        Initialize the seaborn adapter.

        Args:
            theme: The TufteTheme to apply to figures.
        """
        super().__init__(theme)
        if not HAS_SEABORN:
            raise ImportError("seaborn is required for SeabornAdapter")
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for SeabornAdapter (seaborn backend)"
            )
        self.label_positioner = LabelPositioner()

    def apply_theme(
        self, figure: "matplotlib.figure.Figure", theme: TufteTheme
    ) -> "matplotlib.figure.Figure":
        """
        Apply Tufte theme to a seaborn figure.

        This method applies seaborn styling context and then applies
        typography settings, colors, and other theme parameters.

        Args:
            figure: The matplotlib Figure object (seaborn uses matplotlib backend).
            theme: The TufteTheme containing styling parameters.

        Returns:
            The styled matplotlib Figure object.
        """
        # Set seaborn style to white (minimal)
        sns.set_style("white")

        # Set seaborn context for font sizes
        sns.set_context("notebook", font_scale=1.0)

        # Apply theme to all axes (similar to matplotlib adapter)
        for ax in figure.axes:
            # Set background color
            ax.set_facecolor(theme.background_color)
            figure.patch.set_facecolor(theme.background_color)

            # Apply typography
            if ax.get_title():
                ax.title.set_fontsize(theme.title_size)
                ax.title.set_fontfamily(theme.font_family)
                ax.title.set_color(theme.text_color)

            ax.xaxis.label.set_fontsize(theme.label_size)
            ax.xaxis.label.set_fontfamily(theme.font_family)
            ax.xaxis.label.set_color(theme.text_color)

            ax.yaxis.label.set_fontsize(theme.label_size)
            ax.yaxis.label.set_fontfamily(theme.font_family)
            ax.yaxis.label.set_color(theme.text_color)

            # Set tick label properties
            for label in ax.get_xticklabels():
                label.set_fontsize(theme.tick_size)
                label.set_fontfamily(theme.font_family)
                label.set_color(theme.text_color)

            for label in ax.get_yticklabels():
                label.set_fontsize(theme.tick_size)
                label.set_fontfamily(theme.font_family)
                label.set_color(theme.text_color)

            # Set axis colors
            ax.spines["bottom"].set_color(theme.axis_color)
            ax.spines["left"].set_color(theme.axis_color)
            ax.spines["bottom"].set_linewidth(theme.axis_line_width)
            ax.spines["left"].set_linewidth(theme.axis_line_width)

            ax.tick_params(
                axis="both", colors=theme.axis_color, width=theme.axis_line_width
            )

            # Handle grid
            if theme.show_grid:
                ax.grid(True, color=theme.grid_color, linewidth=0.5, alpha=0.5)
            else:
                ax.grid(False)

            # Remove legend if specified
            if theme.remove_legend:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

        return figure

    def remove_chartjunk(
        self, figure: "matplotlib.figure.Figure"
    ) -> "matplotlib.figure.Figure":
        """
        Remove unnecessary visual elements from seaborn figure.

        This uses seaborn's despine function and ensures minimal visual clutter
        following Tufte's principles.

        Args:
            figure: The matplotlib Figure object to modify.

        Returns:
            The modified matplotlib Figure object.
        """
        # Use seaborn's despine to remove top and right spines
        sns.despine(fig=figure, top=True, right=True, left=False, bottom=False)

        for ax in figure.axes:
            # Make bottom and left spines thinner
            ax.spines["bottom"].set_linewidth(self.theme.axis_line_width)
            ax.spines["left"].set_linewidth(self.theme.axis_line_width)

            # Remove tick marks on top and right
            ax.tick_params(top=False, right=False)

        return figure

    def apply_range_frame(
        self, figure: "matplotlib.figure.Figure", data_range: Tuple[float, float]
    ) -> "matplotlib.figure.Figure":
        """
        Set axis limits to match the data range.

        Args:
            figure: The matplotlib Figure object to modify.
            data_range: Tuple of (min, max) values for the data.

        Returns:
            The modified matplotlib Figure object.
        """
        for ax in figure.axes:
            # Apply range frame to y-axis (most common use case)
            ax.set_ylim(data_range[0], data_range[1])

        return figure

    def add_direct_labels(
        self,
        figure: "matplotlib.figure.Figure",
        labels: List[str],
        positions: List[Tuple[float, float]],
    ) -> "matplotlib.figure.Figure":
        """
        Add labels directly on the seaborn plot.

        Args:
            figure: The matplotlib Figure object to modify.
            labels: List of label strings.
            positions: List of (x, y) coordinates for each label.

        Returns:
            The modified matplotlib Figure object.
        """
        if len(labels) != len(positions):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match "
                f"number of positions ({len(positions)})"
            )

        for ax in figure.axes:
            for label, (x, y) in zip(labels, positions):
                ax.text(
                    x,
                    y,
                    label,
                    fontsize=self.theme.label_size,
                    fontfamily=self.theme.font_family,
                    color=self.theme.text_color,
                    verticalalignment="center",
                )

        return figure

    def create_line_plot(
        self, data: "pd.DataFrame", x: str, y: str, **kwargs
    ) -> "matplotlib.figure.Figure":
        """
        Create a styled line plot using seaborn.

        Args:
            data: The pandas DataFrame containing the data.
            x: Column name for x-axis data.
            y: Column name for y-axis data.
            **kwargs: Additional plotting parameters (hue, figsize, etc.).

        Returns:
            The created matplotlib Figure object.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for create_line_plot")

        figsize = kwargs.get("figsize", (8, 5))
        hue = kwargs.get("hue", None)

        fig, ax = plt.subplots(figsize=figsize)

        # Use seaborn's lineplot
        # Ensure palette is in hex format for seaborn
        palette = ensure_hex_palette(self.theme.color_palette)

        if hue is not None:
            sns.lineplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                palette=palette,
                linewidth=self.theme.line_width,
                ax=ax,
            )
        else:
            # When no hue, explicitly set color (seaborn ignores palette without hue)
            sns.lineplot(
                data=data,
                x=x,
                y=y,
                color=palette[0],
                linewidth=self.theme.line_width,
                ax=ax,
            )

        # Apply theme and remove chartjunk
        self.apply_theme(fig, self.theme)
        self.remove_chartjunk(fig)

        return fig

    def create_scatter_plot(
        self, data: "pd.DataFrame", x: str, y: str, **kwargs
    ) -> "matplotlib.figure.Figure":
        """
        Create a styled scatter plot using seaborn.

        Args:
            data: The pandas DataFrame containing the data.
            x: Column name for x-axis data.
            y: Column name for y-axis data.
            **kwargs: Additional plotting parameters (hue, figsize, etc.).

        Returns:
            The created matplotlib Figure object.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for create_scatter_plot")

        figsize = kwargs.get("figsize", (8, 5))
        hue = kwargs.get("hue", None)

        fig, ax = plt.subplots(figsize=figsize)

        # Use seaborn's scatterplot
        # Ensure palette is in hex format for seaborn
        palette = ensure_hex_palette(self.theme.color_palette)

        if hue is not None:
            sns.scatterplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                palette=palette,
                s=30,
                alpha=0.7,
                ax=ax,
            )
        else:
            # When no hue, explicitly set color (seaborn ignores palette without hue)
            sns.scatterplot(
                data=data,
                x=x,
                y=y,
                color=palette[0],
                s=30,
                alpha=0.7,
                ax=ax,
            )

        # Apply theme and remove chartjunk
        self.apply_theme(fig, self.theme)
        self.remove_chartjunk(fig)

        return fig

    def create_bar_plot(
        self, data: "pd.DataFrame", x: str, y: str, **kwargs
    ) -> "matplotlib.figure.Figure":
        """
        Create a styled bar plot using seaborn.

        Args:
            data: The pandas DataFrame containing the data.
            x: Column name for x-axis data (categories).
            y: Column name for y-axis data (values).
            **kwargs: Additional plotting parameters (figsize, etc.).

        Returns:
            The created matplotlib Figure object.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for create_bar_plot")

        figsize = kwargs.get("figsize", (8, 5))

        fig, ax = plt.subplots(figsize=figsize)

        # Use seaborn's barplot
        hex_color = self.theme.color_palette[0]
        color = normalize_color(hex_color)
        sns.barplot(
            data=data,
            x=x,
            y=y,
            color=color,
            ax=ax,
        )

        # Apply theme and remove chartjunk
        self.apply_theme(fig, self.theme)
        self.remove_chartjunk(fig)

        return fig

    def create_histogram(
        self, data: "pd.DataFrame", column: str, **kwargs
    ) -> "matplotlib.figure.Figure":
        """
        Create a styled histogram using seaborn.

        Args:
            data: The pandas DataFrame containing the data.
            column: Column name for the data to histogram.
            **kwargs: Additional plotting parameters (bins, figsize, etc.).

        Returns:
            The created matplotlib Figure object.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for create_histogram")

        figsize = kwargs.get("figsize", (8, 5))
        bins = kwargs.get("bins", 30)

        fig, ax = plt.subplots(figsize=figsize)

        # Use seaborn's histplot
        hex_color = self.theme.color_palette[0]
        color = normalize_color(hex_color)
        sns.histplot(
            data=data,
            x=column,
            bins=bins,
            color=color,
            alpha=0.7,
            ax=ax,
        )

        # Apply theme and remove chartjunk
        self.apply_theme(fig, self.theme)
        self.remove_chartjunk(fig)

        return fig

    def save(self, figure: "matplotlib.figure.Figure", filepath: str, **kwargs) -> None:
        """
        Export seaborn figure to a file.

        Automatically detects format from file extension and configures
        font embedding for PDF and SVG formats to ensure consistent rendering.

        Args:
            figure: The matplotlib Figure object to save.
            filepath: Path where the file should be saved.
            **kwargs: Additional save parameters (dpi, bbox_inches, format, etc.).

        Supported formats: PNG, PDF, SVG, JPG/JPEG
        """
        import os

        dpi = kwargs.get("dpi", 150)
        bbox_inches = kwargs.get("bbox_inches", "tight")

        # Detect format from file extension
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        # Map extensions to matplotlib format names
        format_map = {
            ".png": "png",
            ".pdf": "pdf",
            ".svg": "svg",
            ".jpg": "jpg",
            ".jpeg": "jpg",
        }

        # Get format from extension or kwargs
        fmt = kwargs.get("format", format_map.get(ext, "png"))

        # Configure font embedding for PDF and SVG
        save_kwargs = {
            "dpi": dpi,
            "bbox_inches": bbox_inches,
            "format": fmt,
        }

        # Enable font embedding for PDF
        if fmt == "pdf":
            # Use Type 42 (TrueType) fonts for better compatibility
            save_kwargs["metadata"] = {
                "Creator": "TuftePlots",
                "Producer": "matplotlib",
            }
            # Ensure fonts are embedded by using the pdf backend properly
            import matplotlib

            matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts

        # Enable font embedding for SVG
        elif fmt == "svg":
            # Embed fonts as text for maximum compatibility
            import matplotlib

            matplotlib.rcParams["svg.fonttype"] = (
                "none"  # Embed fonts as text (not paths)
            )

        figure.savefig(filepath, **save_kwargs)
        logger.info("Figure saved to %s (format: %s)", filepath, fmt)

    def show(self, figure: "matplotlib.figure.Figure") -> None:
        """
        Display seaborn figure in the current environment.

        Args:
            figure: The matplotlib Figure object to display.
        """
        plt.show()

    def enable_direct_labeling(
        self, figure: "matplotlib.figure.Figure"
    ) -> "matplotlib.figure.Figure":
        """
        Enable direct labeling for line plots by extracting endpoints and positioning labels.

        This method automatically:
        1. Extracts line endpoints from the figure
        2. Calculates non-overlapping label positions using LabelPositioner
        3. Adds direct labels at calculated positions
        4. Removes the legend

        Args:
            figure: The matplotlib Figure object to modify.

        Returns:
            The modified matplotlib Figure object with direct labels.
        """
        for ax in figure.axes:
            # Get legend to extract proper labels (seaborn creates empty lines for legend)
            legend = ax.get_legend()
            legend_labels = []
            if legend is not None:
                legend_labels = [text.get_text() for text in legend.get_texts()]

            # Extract line data
            lines = ax.get_lines()
            if not lines:
                continue

            # Filter lines with actual data
            data_lines = [line for line in lines if len(line.get_xdata()) > 0]

            if not data_lines:
                continue

            endpoints = []
            labels = []

            for i, line in enumerate(data_lines):
                # Get line data
                xdata = line.get_xdata()
                ydata = line.get_ydata()

                # Get the rightmost point (endpoint)
                endpoint = (xdata[-1], ydata[-1])
                endpoints.append(endpoint)

                # Try to get label from legend first (for seaborn compatibility)
                if i < len(legend_labels):
                    labels.append(legend_labels[i])
                else:
                    # Get label from line (if it has one)
                    label = line.get_label()
                    if label and not label.startswith("_"):
                        labels.append(label)
                    else:
                        labels.append(f"Series {len(labels) + 1}")

            if not endpoints or not labels:
                continue

            # Get axis bounds for label positioning
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            bounds = (xlim[0], xlim[1], ylim[0], ylim[1])

            # Extract data elements (line paths) to avoid label overlap with data
            data_elements = []
            for line in data_lines:
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                # Create list of (x, y) points for this line
                line_points = list(zip(xdata, ydata))
                data_elements.append(line_points)

            # Calculate non-overlapping positions
            positions = self.label_positioner.calculate_positions(
                endpoints, labels, bounds, data_elements=data_elements
            )

            # Add direct labels
            self.add_direct_labels(figure, labels, positions)

            # Remove legend
            if legend is not None:
                legend.remove()

        return figure
