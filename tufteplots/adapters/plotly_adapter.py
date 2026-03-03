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

class PlotlyAdapter(BackendAdapter):
    """
    Adapter for plotly figures.

    This adapter implements Tufte styling for plotly figures, handling
    layout updates, template settings, and trace styling.
    """

    def __init__(self, theme: TufteTheme):
        """
        Initialize the plotly adapter.

        Args:
            theme: The TufteTheme to apply to figures.
        """
        super().__init__(theme)
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for PlotlyAdapter")
        self.label_positioner = LabelPositioner()

    def apply_theme(self, figure: "go.Figure", theme: TufteTheme) -> "go.Figure":
        """
        Apply Tufte theme to a plotly figure.

        This method applies typography settings, colors, and other theme
        parameters to the figure layout.

        Args:
            figure: The plotly Figure object to style.
            theme: The TufteTheme containing styling parameters.

        Returns:
            The styled plotly Figure object.
        """
        # Convert 'transparent' to plotly-compatible format
        plot_bgcolor = theme.background_color
        if plot_bgcolor == "transparent":
            plot_bgcolor = "rgba(0,0,0,0)"

        paper_bgcolor = theme.background_color
        if paper_bgcolor == "transparent":
            paper_bgcolor = "rgba(0,0,0,0)"

        # Apply layout updates for Tufte styling
        figure.update_layout(
            # Template and background
            template="simple_white",
            plot_bgcolor=plot_bgcolor,
            paper_bgcolor=paper_bgcolor,
            # Typography
            font=dict(
                family=theme.font_family,
                size=theme.tick_size,
                color=theme.text_color,
            ),
            # Title styling
            title_font=dict(
                family=theme.font_family,
                size=theme.title_size,
                color=theme.text_color,
            ),
            # Axis labels
            xaxis=dict(
                title_font=dict(
                    family=theme.font_family,
                    size=theme.label_size,
                    color=theme.text_color,
                ),
                tickfont=dict(
                    family=theme.font_family,
                    size=theme.tick_size,
                    color=theme.text_color,
                ),
                linecolor=theme.axis_color,
                linewidth=theme.axis_line_width,
                showgrid=theme.show_grid,
                gridcolor=theme.grid_color if theme.show_grid else None,
            ),
            yaxis=dict(
                title_font=dict(
                    family=theme.font_family,
                    size=theme.label_size,
                    color=theme.text_color,
                ),
                tickfont=dict(
                    family=theme.font_family,
                    size=theme.tick_size,
                    color=theme.text_color,
                ),
                linecolor=theme.axis_color,
                linewidth=theme.axis_line_width,
                showgrid=theme.show_grid,
                gridcolor=theme.grid_color if theme.show_grid else None,
            ),
            # Remove legend if specified
            showlegend=not theme.remove_legend,
        )

        # Update trace colors to match theme palette
        for i, trace in enumerate(figure.data):
            hex_color = theme.color_palette[i % len(theme.color_palette)]
            # Ensure color is in hex format for plotly
            color = normalize_color(hex_color)
            if hasattr(trace, "line"):
                trace.line.color = color
                trace.line.width = theme.line_width
            if hasattr(trace, "marker"):
                trace.marker.color = color

        return figure

    def remove_chartjunk(self, figure: "go.Figure") -> "go.Figure":
        """
        Remove unnecessary visual elements from plotly figure.

        This removes grid lines, ensures minimal visual clutter, and
        removes top and right axis lines following Tufte's principles.

        Args:
            figure: The plotly Figure object to modify.

        Returns:
            The modified plotly Figure object.
        """
        # Update layout to remove chartjunk
        figure.update_xaxes(
            showgrid=False,
            zeroline=False,
            mirror=False,
            showline=True,
            linecolor=self.theme.axis_color,
            linewidth=self.theme.axis_line_width,
        )

        figure.update_yaxes(
            showgrid=False,
            zeroline=False,
            mirror=False,
            showline=True,
            linecolor=self.theme.axis_color,
            linewidth=self.theme.axis_line_width,
        )

        return figure

    def apply_range_frame(
        self, figure: "go.Figure", data_range: Tuple[float, float]
    ) -> "go.Figure":
        """
        Set axis limits to match the data range.

        Args:
            figure: The plotly Figure object to modify.
            data_range: Tuple of (min, max) values for the data.

        Returns:
            The modified plotly Figure object.
        """
        # Apply range frame to y-axis (most common use case)
        figure.update_yaxes(range=[data_range[0], data_range[1]])

        return figure

    def add_direct_labels(
        self,
        figure: "go.Figure",
        labels: List[str],
        positions: List[Tuple[float, float]],
    ) -> "go.Figure":
        """
        Add labels directly on the plotly plot.

        Args:
            figure: The plotly Figure object to modify.
            labels: List of label strings.
            positions: List of (x, y) coordinates for each label.

        Returns:
            The modified plotly Figure object.
        """
        if len(labels) != len(positions):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match "
                f"number of positions ({len(positions)})"
            )

        # Add annotations for direct labels
        annotations = []
        for label, (x, y) in zip(labels, positions):
            annotations.append(
                dict(
                    x=x,
                    y=y,
                    text=label,
                    showarrow=False,
                    font=dict(
                        family=self.theme.font_family,
                        size=self.theme.label_size,
                        color=self.theme.text_color,
                    ),
                    xanchor="left",
                    yanchor="middle",
                )
            )

        figure.update_layout(annotations=annotations)

        return figure

    def create_line_plot(
        self, data: "pd.DataFrame", x: str, y: str, **kwargs
    ) -> "go.Figure":
        """
        Create a styled line plot using plotly.

        Args:
            data: The pandas DataFrame containing the data.
            x: Column name for x-axis data.
            y: Column name for y-axis data.
            **kwargs: Additional plotting parameters (hue, figsize, etc.).

        Returns:
            The created plotly Figure object.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for create_line_plot")

        hue = kwargs.get("hue", None)
        fig = go.Figure()

        if hue is not None:
            # Plot multiple lines grouped by hue
            for i, (name, group) in enumerate(data.groupby(hue)):
                hex_color = self.theme.color_palette[i % len(self.theme.color_palette)]
                color = normalize_color(hex_color)
                fig.add_trace(
                    go.Scatter(
                        x=group[x],
                        y=group[y],
                        mode="lines",
                        name=str(name),
                        line=dict(color=color, width=self.theme.line_width),
                    )
                )
        else:
            # Single line plot
            hex_color = self.theme.color_palette[0]
            color = normalize_color(hex_color)
            fig.add_trace(
                go.Scatter(
                    x=data[x],
                    y=data[y],
                    mode="lines",
                    line=dict(color=color, width=self.theme.line_width),
                )
            )

        # Apply theme and remove chartjunk
        self.apply_theme(fig, self.theme)
        self.remove_chartjunk(fig)

        return fig

    def create_scatter_plot(
        self, data: "pd.DataFrame", x: str, y: str, **kwargs
    ) -> "go.Figure":
        """
        Create a styled scatter plot using plotly.

        Args:
            data: The pandas DataFrame containing the data.
            x: Column name for x-axis data.
            y: Column name for y-axis data.
            **kwargs: Additional plotting parameters (hue, figsize, etc.).

        Returns:
            The created plotly Figure object.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for create_scatter_plot")

        hue = kwargs.get("hue", None)
        fig = go.Figure()

        if hue is not None:
            # Plot multiple series grouped by hue
            for i, (name, group) in enumerate(data.groupby(hue)):
                hex_color = self.theme.color_palette[i % len(self.theme.color_palette)]
                color = normalize_color(hex_color)
                fig.add_trace(
                    go.Scatter(
                        x=group[x],
                        y=group[y],
                        mode="markers",
                        name=str(name),
                        marker=dict(color=color, size=8, opacity=0.7),
                    )
                )
        else:
            # Single scatter plot
            hex_color = self.theme.color_palette[0]
            color = normalize_color(hex_color)
            fig.add_trace(
                go.Scatter(
                    x=data[x],
                    y=data[y],
                    mode="markers",
                    marker=dict(color=color, size=8, opacity=0.7),
                )
            )

        # Apply theme and remove chartjunk
        self.apply_theme(fig, self.theme)
        self.remove_chartjunk(fig)

        return fig

    def create_bar_plot(
        self, data: "pd.DataFrame", x: str, y: str, **kwargs
    ) -> "go.Figure":
        """
        Create a styled bar plot using plotly.

        Args:
            data: The pandas DataFrame containing the data.
            x: Column name for x-axis data (categories).
            y: Column name for y-axis data (values).
            **kwargs: Additional plotting parameters.

        Returns:
            The created plotly Figure object.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for create_bar_plot")

        fig = go.Figure()

        hex_color = self.theme.color_palette[0]
        color = normalize_color(hex_color)
        fig.add_trace(
            go.Bar(
                x=data[x],
                y=data[y],
                marker=dict(
                    color=color,
                    line=dict(width=0),  # No borders
                ),
            )
        )

        # Apply theme and remove chartjunk
        self.apply_theme(fig, self.theme)
        self.remove_chartjunk(fig)

        return fig

    def create_histogram(
        self, data: "pd.DataFrame", column: str, **kwargs
    ) -> "go.Figure":
        """
        Create a styled histogram using plotly.

        Args:
            data: The pandas DataFrame containing the data.
            column: Column name for the data to histogram.
            **kwargs: Additional plotting parameters (bins, etc.).

        Returns:
            The created plotly Figure object.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for create_histogram")

        nbins = kwargs.get("bins", 30)
        fig = go.Figure()

        hex_color = self.theme.color_palette[0]
        color = normalize_color(hex_color)
        fig.add_trace(
            go.Histogram(
                x=data[column],
                nbinsx=nbins,
                marker=dict(
                    color=color,
                    line=dict(width=0),  # No borders
                ),
                opacity=0.7,
            )
        )

        # Apply theme and remove chartjunk
        self.apply_theme(fig, self.theme)
        self.remove_chartjunk(fig)

        return fig

    def save(self, figure: "go.Figure", filepath: str, **kwargs) -> None:
        """
        Export plotly figure to a file.

        Automatically detects format from file extension and configures
        font embedding for PDF and SVG formats to ensure consistent rendering.

        Args:
            figure: The plotly Figure object to save.
            filepath: Path where the file should be saved.
            **kwargs: Additional save parameters (width, height, scale, format, etc.).

        Supported formats: PNG, PDF, SVG, JPG/JPEG, HTML
        """
        import os

        # Determine format from file extension
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        width = kwargs.get("width", 800)
        height = kwargs.get("height", 500)
        scale = kwargs.get("scale", 2)

        # Map extensions to plotly format names
        format_map = {
            ".png": "png",
            ".pdf": "pdf",
            ".svg": "svg",
            ".jpg": "jpg",
            ".jpeg": "jpg",
            ".html": "html",
        }

        # Get format from extension or kwargs
        fmt = kwargs.get("format", format_map.get(ext, "html"))

        if fmt in ["png", "jpg", "jpeg"]:
            figure.write_image(
                filepath, width=width, height=height, scale=scale, format=fmt
            )
        elif fmt == "pdf":
            # For PDF, plotly's write_image uses kaleido which embeds fonts by default
            # We ensure proper font rendering by setting the format explicitly
            figure.write_image(
                filepath,
                width=width,
                height=height,
                format="pdf",
                scale=scale,
            )
        elif fmt == "svg":
            # For SVG, plotly's write_image embeds fonts as text by default
            # This ensures consistent rendering across different viewers
            figure.write_image(
                filepath,
                width=width,
                height=height,
                format="svg",
                scale=scale,
            )
        elif fmt == "html":
            # HTML format includes all fonts inline
            figure.write_html(filepath)
        else:
            # Default to HTML for unknown formats
            logger.warning("Unknown format '%s', defaulting to HTML", ext)
            figure.write_html(filepath)

        logger.info("Figure saved to %s (format: %s)", filepath, fmt)

    def show(self, figure: "go.Figure") -> None:
        """
        Display plotly figure in the current environment.

        Args:
            figure: The plotly Figure object to display.
        """
        figure.show()

    def enable_direct_labeling(self, figure: "go.Figure") -> "go.Figure":
        """
        Enable direct labeling for line plots by extracting endpoints and positioning labels.

        This method automatically:
        1. Extracts line endpoints from the figure traces
        2. Calculates non-overlapping label positions using LabelPositioner
        3. Adds direct labels at calculated positions
        4. Removes the legend

        Args:
            figure: The plotly Figure object to modify.

        Returns:
            The modified plotly Figure object with direct labels.
        """
        endpoints = []
        labels = []

        # Extract data from traces
        for trace in figure.data:
            # Only process line traces
            if hasattr(trace, "mode") and "lines" in trace.mode:
                if hasattr(trace, "x") and hasattr(trace, "y"):
                    x_data = trace.x
                    y_data = trace.y

                    if len(x_data) == 0 or len(y_data) == 0:
                        continue

                    # Get the rightmost point (endpoint)
                    endpoint = (x_data[-1], y_data[-1])
                    endpoints.append(endpoint)

                    # Get label from trace name
                    if hasattr(trace, "name") and trace.name:
                        labels.append(trace.name)
                    else:
                        labels.append(f"Series {len(labels) + 1}")

        if not endpoints or not labels:
            return figure

        # Get axis bounds for label positioning
        # Extract from layout or calculate from data
        if figure.layout.xaxis.range:
            x_min, x_max = figure.layout.xaxis.range
        else:
            all_x = [x for trace in figure.data if hasattr(trace, "x") for x in trace.x]
            x_min, x_max = min(all_x), max(all_x)

        if figure.layout.yaxis.range:
            y_min, y_max = figure.layout.yaxis.range
        else:
            all_y = [y for trace in figure.data if hasattr(trace, "y") for y in trace.y]
            y_min, y_max = min(all_y), max(all_y)

        bounds = (x_min, x_max, y_min, y_max)

        # Extract data elements (line paths) to avoid label overlap with data
        data_elements = []
        for trace in figure.data:
            if hasattr(trace, "mode") and "lines" in trace.mode:
                if hasattr(trace, "x") and hasattr(trace, "y"):
                    x_data = trace.x
                    y_data = trace.y
                    if len(x_data) > 0 and len(y_data) > 0:
                        # Create list of (x, y) points for this line
                        line_points = list(zip(x_data, y_data))
                        data_elements.append(line_points)

        # Calculate non-overlapping positions
        positions = self.label_positioner.calculate_positions(
            endpoints, labels, bounds, data_elements=data_elements
        )

        # Add direct labels
        self.add_direct_labels(figure, labels, positions)

        # Remove legend
        figure.update_layout(showlegend=False)

        return figure
