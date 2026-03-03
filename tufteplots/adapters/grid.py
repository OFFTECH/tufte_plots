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

def calculate_grid_dimensions(n_groups: int) -> Tuple[int, int]:
    """
    Calculate optimal grid dimensions for small multiples.

    This function calculates the number of rows and columns needed to display
    n_groups subplots, minimizing empty cells while preferring wider layouts.

    Args:
        n_groups: Number of groups/subplots to display.

    Returns:
        Tuple of (rows, cols) for the grid layout.

    Example:
        >>> calculate_grid_dimensions(6)
        (2, 3)
        >>> calculate_grid_dimensions(7)
        (3, 3)
    """
    import math

    if n_groups <= 0:
        return (0, 0)
    if n_groups == 1:
        return (1, 1)

    # Calculate number of columns (prefer wider layouts)
    # Use ceiling of square root as starting point
    cols = math.ceil(math.sqrt(n_groups))

    # Calculate rows needed
    rows = math.ceil(n_groups / cols)

    # Optimize: try to reduce rows if possible
    # Check if we can fit in fewer rows with more columns
    if rows > 1:
        min_cols_needed = math.ceil(n_groups / (rows - 1))
        if min_cols_needed <= cols + 1:
            rows = rows - 1
            cols = min_cols_needed

    return (rows, cols)

def small_multiples(
    data: "pd.DataFrame",
    x: str,
    y: str,
    facet_by: str,
    plot_type: str = "line",
    backend: str = "matplotlib",
    theme: TufteTheme = None,
    **kwargs,
) -> FigureType:
    """
    Create a grid of small multiple plots.

    This function creates a grid of subplots, one for each unique value in the
    facet_by column, with shared axes for easy comparison. Each subplot is
    styled according to Tufte principles.

    Args:
        data: The pandas DataFrame containing the data.
        x: Column name for x-axis data.
        y: Column name for y-axis data.
        facet_by: Column name to group data by (creates one subplot per group).
        plot_type: Type of plot to create ("line", "scatter", "bar", "histogram").
        backend: Backend to use ("matplotlib", "plotly", "seaborn").
        theme: Optional TufteTheme to use. If None, uses default theme.
        **kwargs: Additional plotting parameters (figsize, etc.).

    Returns:
        The created figure object with small multiples.

    Raises:
        ValueError: If backend is not supported or plot_type is invalid.
        KeyError: If required columns are missing from the DataFrame.

    Example:
        >>> import pandas as pd
        >>> from tufteplots.adapters import small_multiples
        >>> data = pd.DataFrame({
        ...     'x': range(100),
        ...     'y': range(100),
        ...     'category': ['A'] * 50 + ['B'] * 50
        ... })
        >>> fig = small_multiples(data, 'x', 'y', 'category', plot_type='line')
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for small_multiples")

    # Validate required columns
    required_cols = [x, y, facet_by]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # Validate plot type
    valid_plot_types = ["line", "scatter", "bar", "histogram"]
    if plot_type not in valid_plot_types:
        raise ValueError(
            f"Invalid plot_type '{plot_type}'. " f"Supported types: {valid_plot_types}"
        )

    # Validate backend
    valid_backends = ["matplotlib", "plotly", "seaborn"]
    if backend not in valid_backends:
        raise ValueError(
            f"Unsupported backend '{backend}'. " f"Supported backends: {valid_backends}"
        )

    # Use default theme if none provided
    if theme is None:
        theme = TufteTheme()

    # Get unique groups
    groups = data[facet_by].unique()
    n_groups = len(groups)

    # Calculate grid dimensions
    rows, cols = calculate_grid_dimensions(n_groups)

    # Create figure based on backend
    if backend == "matplotlib" or backend == "seaborn":
        return _create_matplotlib_small_multiples(
            data,
            x,
            y,
            facet_by,
            groups,
            rows,
            cols,
            plot_type,
            theme,
            backend,
            **kwargs,
        )
    elif backend == "plotly":
        return _create_plotly_small_multiples(
            data, x, y, facet_by, groups, rows, cols, plot_type, theme, **kwargs
        )

def _create_matplotlib_small_multiples(
    data: "pd.DataFrame",
    x: str,
    y: str,
    facet_by: str,
    groups: Any,
    rows: int,
    cols: int,
    plot_type: str,
    theme: TufteTheme,
    backend: str,
    **kwargs,
) -> "matplotlib.figure.Figure":
    """
    Create small multiples using matplotlib or seaborn backend.

    Args:
        data: The pandas DataFrame containing the data.
        x: Column name for x-axis data.
        y: Column name for y-axis data.
        facet_by: Column name to group data by.
        groups: Array of unique group values.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        plot_type: Type of plot to create.
        theme: TufteTheme to apply.
        backend: "matplotlib" or "seaborn".
        **kwargs: Additional plotting parameters.

    Returns:
        The created matplotlib Figure object.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for matplotlib/seaborn backend")

    # Calculate figure size
    subplot_width = kwargs.get("subplot_width", 4)
    subplot_height = kwargs.get("subplot_height", 3)
    figsize = (cols * subplot_width, rows * subplot_height)

    # Create subplots with shared axes
    fig, axes = plt.subplots(
        rows, cols, figsize=figsize, sharex=True, sharey=True, squeeze=False
    )

    # Check if seaborn is needed
    if backend == "seaborn" and not HAS_SEABORN:
        raise ImportError("seaborn is required for seaborn backend")

    # Plot each group
    for idx, group_value in enumerate(groups):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Filter data for this group
        group_data = data[data[facet_by] == group_value]

        # Create plot based on type
        hex_color = theme.color_palette[0]
        color = hex_to_rgb_normalized(hex_color)

        if plot_type == "line":
            ax.plot(
                group_data[x],
                group_data[y],
                color=color,
                linewidth=theme.line_width,
            )
        elif plot_type == "scatter":
            ax.scatter(
                group_data[x],
                group_data[y],
                color=color,
                s=30,
                alpha=0.7,
            )
        elif plot_type == "bar":
            ax.bar(
                group_data[x],
                group_data[y],
                color=color,
                edgecolor="none",
            )
        elif plot_type == "histogram":
            bins = kwargs.get("bins", 30)
            ax.hist(
                group_data[y],
                bins=bins,
                color=color,
                edgecolor="none",
                alpha=0.7,
            )

        # Set subplot title with group label
        ax.set_title(
            str(group_value), fontsize=theme.title_size, fontfamily=theme.font_family
        )

        # Apply theme to this subplot
        ax.set_facecolor(theme.background_color)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(theme.axis_line_width)
        ax.spines["left"].set_linewidth(theme.axis_line_width)
        ax.spines["bottom"].set_color(theme.axis_color)
        ax.spines["left"].set_color(theme.axis_color)

        # Set tick parameters
        ax.tick_params(
            axis="both",
            colors=theme.axis_color,
            width=theme.axis_line_width,
            labelsize=theme.tick_size,
        )

        # Handle grid
        if theme.show_grid:
            ax.grid(True, color=theme.grid_color, linewidth=0.5, alpha=0.5)
        else:
            ax.grid(False)

    # Hide unused subplots
    for idx in range(len(groups), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)

    # Set overall figure background
    fig.patch.set_facecolor(theme.background_color)

    # Adjust layout
    plt.tight_layout()

    return fig

def _create_plotly_small_multiples(
    data: "pd.DataFrame",
    x: str,
    y: str,
    facet_by: str,
    groups: Any,
    rows: int,
    cols: int,
    plot_type: str,
    theme: TufteTheme,
    **kwargs,
) -> "go.Figure":
    """
    Create small multiples using plotly backend.

    Args:
        data: The pandas DataFrame containing the data.
        x: Column name for x-axis data.
        y: Column name for y-axis data.
        facet_by: Column name to group data by.
        groups: Array of unique group values.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        plot_type: Type of plot to create.
        theme: TufteTheme to apply.
        **kwargs: Additional plotting parameters.

    Returns:
        The created plotly Figure object.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotly backend")

    from plotly.subplots import make_subplots

    # Create subplot titles
    subplot_titles = [str(group) for group in groups]

    # Create subplots with shared axes
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )

    # Plot each group
    for idx, group_value in enumerate(groups):
        row = (idx // cols) + 1  # plotly uses 1-based indexing
        col = (idx % cols) + 1

        # Filter data for this group
        group_data = data[data[facet_by] == group_value]

        # Create trace based on plot type
        hex_color = theme.color_palette[0]
        color = normalize_color(hex_color)

        trace = None
        if plot_type == "line":
            trace = go.Scatter(
                x=group_data[x],
                y=group_data[y],
                mode="lines",
                line=dict(color=color, width=theme.line_width),
                showlegend=False,
            )
        elif plot_type == "scatter":
            trace = go.Scatter(
                x=group_data[x],
                y=group_data[y],
                mode="markers",
                marker=dict(color=color, size=8, opacity=0.7),
                showlegend=False,
            )
        elif plot_type == "bar":
            trace = go.Bar(
                x=group_data[x],
                y=group_data[y],
                marker=dict(color=color, line=dict(width=0)),
                showlegend=False,
            )
        elif plot_type == "histogram":
            bins = kwargs.get("bins", 30)
            trace = go.Histogram(
                x=group_data[y],
                nbinsx=bins,
                marker=dict(color=color, line=dict(width=0)),
                opacity=0.7,
                showlegend=False,
            )

        if trace is not None:
            fig.add_trace(trace, row=row, col=col)

    # Apply theme to layout
    plot_bgcolor = theme.background_color
    if plot_bgcolor == "transparent":
        plot_bgcolor = "rgba(0,0,0,0)"

    paper_bgcolor = theme.background_color
    if paper_bgcolor == "transparent":
        paper_bgcolor = "rgba(0,0,0,0)"

    fig.update_layout(
        template="simple_white",
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font=dict(
            family=theme.font_family,
            size=theme.tick_size,
            color=theme.text_color,
        ),
        showlegend=False,
    )

    # Update all axes
    fig.update_xaxes(
        showgrid=theme.show_grid,
        gridcolor=theme.grid_color if theme.show_grid else None,
        linecolor=theme.axis_color,
        linewidth=theme.axis_line_width,
        mirror=False,
        showline=True,
    )

    fig.update_yaxes(
        showgrid=theme.show_grid,
        gridcolor=theme.grid_color if theme.show_grid else None,
        linecolor=theme.axis_color,
        linewidth=theme.axis_line_width,
        mirror=False,
        showline=True,
    )

    return fig

