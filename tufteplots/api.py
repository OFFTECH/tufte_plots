"""
Public API functions for TuftePlots.

This module contains the main user-facing functions for creating and styling
Tufte-style visualizations across different plotting backends.
"""

from typing import Any, Optional
import logging

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.figure

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

from tufteplots.theme import TufteTheme, ThemeManager
from tufteplots.adapters import MatplotlibAdapter, PlotlyAdapter, SeabornAdapter
from tufteplots.validators import ConfigValidator

logger = logging.getLogger("tufteplots")

# Type alias for figure types
FigureType = Any


def _detect_backend(figure: Any) -> str:
    """
    Auto-detect the backend from the figure type.

    Args:
        figure: The figure object to inspect.

    Returns:
        The detected backend name ('matplotlib', 'plotly', or 'seaborn').

    Raises:
        TypeError: If the figure type cannot be determined.
    """
    figure_type = type(figure).__module__ + "." + type(figure).__name__

    if HAS_MATPLOTLIB and isinstance(figure, matplotlib.figure.Figure):
        return "matplotlib"
    elif (
        HAS_PLOTLY
        and "plotly" in figure_type.lower()
        and "figure" in figure_type.lower()
    ):
        return "plotly"
    elif HAS_SEABORN and "seaborn" in figure_type.lower():
        return "seaborn"
    else:
        raise TypeError(
            f"Cannot detect backend for figure type {figure_type}. "
            f"Supported types: matplotlib.figure.Figure, plotly.graph_objects.Figure"
        )


def apply_tufte_style(
    figure: FigureType,
    backend: Optional[str] = None,
    theme: Optional[TufteTheme] = None,
    direct_labels: bool = False,
    **kwargs,
) -> FigureType:
    """
    Apply Tufte styling to an existing figure.

    This function automatically detects the backend from the figure type
    (or uses the specified backend) and applies Tufte styling principles
    including chartjunk removal, typography settings, and optional direct labeling.

    Args:
        figure: A matplotlib, plotly, or seaborn figure object.
        backend: Override auto-detection of backend. If None, backend is
                auto-detected from figure type. Valid values: 'matplotlib',
                'plotly', 'seaborn'.
        theme: Custom theme configuration. If None, uses default TufteTheme.
        direct_labels: Enable direct labeling (removes legend and adds labels
                      at data endpoints). Only applicable to line plots.
        **kwargs: Additional styling parameters (currently unused, reserved
                 for future extensions).

    Returns:
        The styled figure object (same type as input).

    Raises:
        ValueError: If backend is not supported.
        TypeError: If figure type doesn't match the specified backend.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 2, 3])
        >>> fig = apply_tufte_style(fig)
        >>> plt.show()
    """
    # Auto-detect backend if not specified
    if backend is None:
        backend = _detect_backend(figure)
    else:
        # Validate backend
        ConfigValidator.validate_backend(backend)
        # Validate figure type matches backend
        ConfigValidator.validate_figure(figure, backend)

    # Use default theme if none provided
    if theme is None:
        theme = TufteTheme()

    # Create appropriate adapter
    if backend == "matplotlib":
        adapter = MatplotlibAdapter(theme)
    elif backend == "plotly":
        adapter = PlotlyAdapter(theme)
    elif backend == "seaborn":
        adapter = SeabornAdapter(theme)
    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    # Apply theme and remove chartjunk
    figure = adapter.apply_theme(figure, theme)
    figure = adapter.remove_chartjunk(figure)

    # Apply direct labeling if requested
    if direct_labels:
        figure = adapter.enable_direct_labeling(figure)

    return figure


def tufte_line_plot(
    data: "pd.DataFrame",
    x: str,
    y: str,
    hue: Optional[str] = None,
    backend: str = "matplotlib",
    direct_labels: bool = True,
    theme: Optional[TufteTheme] = None,
    **kwargs,
) -> FigureType:
    """
    Create a Tufte-styled line plot.

    This function creates a line plot with Tufte styling applied by default,
    including minimal borders, clean typography, and optional direct labeling
    for multiple series.

    Args:
        data: The pandas DataFrame containing the data.
        x: Column name for x-axis data.
        y: Column name for y-axis data.
        hue: Optional column name for grouping data into multiple series.
             Each unique value creates a separate line.
        backend: Backend to use for rendering. Valid values: 'matplotlib',
                'plotly', 'seaborn'. Default is 'matplotlib'.
        direct_labels: Enable direct labeling at series endpoints instead of
                      using a legend. Default is True.
        theme: Custom theme configuration. If None, uses default TufteTheme.
        **kwargs: Additional plotting parameters (figsize, etc.).

    Returns:
        The created figure object (type depends on backend).

    Raises:
        ValueError: If backend is not supported.
        KeyError: If required columns are missing from the DataFrame.
        ImportError: If required backend library is not installed.

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     'x': range(10),
        ...     'y': range(10),
        ...     'category': ['A'] * 5 + ['B'] * 5
        ... })
        >>> fig = tufte_line_plot(data, 'x', 'y', hue='category')
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for tufte_line_plot")

    # Validate backend
    ConfigValidator.validate_backend(backend)

    # Validate required columns
    required_cols = [x, y]
    if hue is not None:
        required_cols.append(hue)
    ConfigValidator.validate_dataframe_columns(data, required_cols)

    # Use default theme if none provided
    if theme is None:
        theme = TufteTheme()

    # Create appropriate adapter
    if backend == "matplotlib":
        adapter = MatplotlibAdapter(theme)
    elif backend == "plotly":
        adapter = PlotlyAdapter(theme)
    elif backend == "seaborn":
        adapter = SeabornAdapter(theme)
    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    # Create line plot
    figure = adapter.create_line_plot(data, x, y, hue=hue, **kwargs)

    # Apply direct labeling if requested and hue is specified
    if direct_labels and hue is not None:
        figure = adapter.enable_direct_labeling(figure)

    return figure


def tufte_scatter_plot(
    data: "pd.DataFrame",
    x: str,
    y: str,
    hue: Optional[str] = None,
    backend: str = "matplotlib",
    show_trend: bool = False,
    theme: Optional[TufteTheme] = None,
    **kwargs,
) -> FigureType:
    """
    Create a Tufte-styled scatter plot.

    This function creates a scatter plot with subtle markers and Tufte styling,
    including minimal borders and clean typography. Optionally adds a trend line
    annotation.

    Args:
        data: The pandas DataFrame containing the data.
        x: Column name for x-axis data.
        y: Column name for y-axis data.
        hue: Optional column name for grouping data by color.
        backend: Backend to use for rendering. Valid values: 'matplotlib',
                'plotly', 'seaborn'. Default is 'matplotlib'.
        show_trend: If True, adds a linear trend line to the plot.
                   Default is False.
        theme: Custom theme configuration. If None, uses default TufteTheme.
        **kwargs: Additional plotting parameters (figsize, etc.).

    Returns:
        The created figure object (type depends on backend).

    Raises:
        ValueError: If backend is not supported.
        KeyError: If required columns are missing from the DataFrame.
        ImportError: If required backend library is not installed.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = pd.DataFrame({
        ...     'x': np.random.randn(100),
        ...     'y': np.random.randn(100)
        ... })
        >>> fig = tufte_scatter_plot(data, 'x', 'y', show_trend=True)
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for tufte_scatter_plot")

    # Validate backend
    ConfigValidator.validate_backend(backend)

    # Validate required columns
    required_cols = [x, y]
    if hue is not None:
        required_cols.append(hue)
    ConfigValidator.validate_dataframe_columns(data, required_cols)

    # Use default theme if none provided
    if theme is None:
        theme = TufteTheme()

    # Create appropriate adapter
    if backend == "matplotlib":
        adapter = MatplotlibAdapter(theme)
    elif backend == "plotly":
        adapter = PlotlyAdapter(theme)
    elif backend == "seaborn":
        adapter = SeabornAdapter(theme)
    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    # Create scatter plot
    figure = adapter.create_scatter_plot(data, x, y, hue=hue, **kwargs)

    # Add trend line if requested
    if show_trend:
        _add_trend_line(figure, data, x, y, backend, theme)

    return figure


def _add_trend_line(
    figure: Any, data: "pd.DataFrame", x: str, y: str, backend: str, theme: TufteTheme
) -> None:
    """
    Add a linear trend line to a scatter plot.

    Args:
        figure: The figure object to modify.
        data: The pandas DataFrame containing the data.
        x: Column name for x-axis data.
        y: Column name for y-axis data.
        backend: The backend being used.
        theme: The theme configuration.
    """
    import numpy as np

    # Calculate linear regression
    x_data = data[x].values
    y_data = data[y].values

    # Remove NaN values
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]

    if len(x_clean) < 2:
        return

    # Fit linear regression
    coeffs = np.polyfit(x_clean, y_clean, 1)
    trend_line = np.poly1d(coeffs)

    # Generate trend line points
    x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_trend = trend_line(x_trend)

    # Add trend line to figure
    if backend == "matplotlib" or backend == "seaborn":
        for ax in figure.axes:
            ax.plot(
                x_trend,
                y_trend,
                "--",
                color=theme.axis_color,
                linewidth=theme.line_width * 0.8,
                alpha=0.7,
                label="Trend",
            )
    elif backend == "plotly":
        figure.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                line=dict(
                    color=theme.axis_color, width=theme.line_width * 0.8, dash="dash"
                ),
                opacity=0.7,
                name="Trend",
                showlegend=False,
            )
        )


def tufte_bar_plot(
    data: "pd.DataFrame",
    x: str,
    y: str,
    backend: str = "matplotlib",
    show_values: bool = True,
    theme: Optional[TufteTheme] = None,
    **kwargs,
) -> FigureType:
    """
    Create a Tufte-styled bar plot.

    This function creates a bar plot with no borders and Tufte styling,
    including minimal visual elements and optional direct value labels on bars.

    Args:
        data: The pandas DataFrame containing the data.
        x: Column name for x-axis data (categories).
        y: Column name for y-axis data (values).
        backend: Backend to use for rendering. Valid values: 'matplotlib',
                'plotly', 'seaborn'. Default is 'matplotlib'.
        show_values: If True, adds value labels directly on top of bars.
                    Default is True.
        theme: Custom theme configuration. If None, uses default TufteTheme.
        **kwargs: Additional plotting parameters (figsize, etc.).

    Returns:
        The created figure object (type depends on backend).

    Raises:
        ValueError: If backend is not supported.
        KeyError: If required columns are missing from the DataFrame.
        ImportError: If required backend library is not installed.

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     'category': ['A', 'B', 'C'],
        ...     'value': [10, 20, 15]
        ... })
        >>> fig = tufte_bar_plot(data, 'category', 'value')
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for tufte_bar_plot")

    # Validate backend
    ConfigValidator.validate_backend(backend)

    # Validate required columns
    ConfigValidator.validate_dataframe_columns(data, [x, y])

    # Use default theme if none provided
    if theme is None:
        theme = TufteTheme()

    # Create appropriate adapter
    if backend == "matplotlib":
        adapter = MatplotlibAdapter(theme)
    elif backend == "plotly":
        adapter = PlotlyAdapter(theme)
    elif backend == "seaborn":
        adapter = SeabornAdapter(theme)
    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    # Create bar plot
    figure = adapter.create_bar_plot(data, x, y, **kwargs)

    # Add value labels if requested
    if show_values:
        _add_bar_value_labels(figure, data, x, y, backend, theme)

    return figure


def _add_bar_value_labels(
    figure: Any, data: "pd.DataFrame", x: str, y: str, backend: str, theme: TufteTheme
) -> None:
    """
    Add value labels on top of bars.

    Args:
        figure: The figure object to modify.
        data: The pandas DataFrame containing the data.
        x: Column name for x-axis data.
        y: Column name for y-axis data.
        backend: The backend being used.
        theme: The theme configuration.
    """
    if backend == "matplotlib" or backend == "seaborn":
        for ax in figure.axes:
            # Get bar containers
            for container in ax.containers:
                ax.bar_label(
                    container,
                    fontsize=theme.tick_size,
                    color=theme.text_color,
                    padding=3,
                )
    elif backend == "plotly":
        # Update traces to show text on bars
        figure.update_traces(
            texttemplate="%{y}",
            textposition="outside",
            textfont=dict(
                family=theme.font_family, size=theme.tick_size, color=theme.text_color
            ),
        )


def tufte_histogram(
    data: "pd.DataFrame",
    column: str,
    backend: str = "matplotlib",
    show_rug: bool = True,
    bins: int = 30,
    theme: Optional[TufteTheme] = None,
    **kwargs,
) -> FigureType:
    """
    Create a Tufte-styled histogram.

    This function creates a histogram with minimal borders and Tufte styling,
    including clean typography and an optional rug plot showing individual
    data points.

    Args:
        data: The pandas DataFrame containing the data.
        column: Column name for the data to histogram.
        backend: Backend to use for rendering. Valid values: 'matplotlib',
                'plotly', 'seaborn'. Default is 'matplotlib'.
        show_rug: If True, adds a rug plot showing individual data points
                 along the x-axis. Default is True.
        bins: Number of bins for the histogram. Default is 30.
        theme: Custom theme configuration. If None, uses default TufteTheme.
        **kwargs: Additional plotting parameters (figsize, etc.).

    Returns:
        The created figure object (type depends on backend).

    Raises:
        ValueError: If backend is not supported.
        KeyError: If required column is missing from the DataFrame.
        ImportError: If required backend library is not installed.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = pd.DataFrame({
        ...     'values': np.random.randn(1000)
        ... })
        >>> fig = tufte_histogram(data, 'values', show_rug=True)
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for tufte_histogram")

    # Validate backend
    ConfigValidator.validate_backend(backend)

    # Validate required columns
    ConfigValidator.validate_dataframe_columns(data, [column])

    # Use default theme if none provided
    if theme is None:
        theme = TufteTheme()

    # Create appropriate adapter
    if backend == "matplotlib":
        adapter = MatplotlibAdapter(theme)
    elif backend == "plotly":
        adapter = PlotlyAdapter(theme)
    elif backend == "seaborn":
        adapter = SeabornAdapter(theme)
    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    # Create histogram
    figure = adapter.create_histogram(data, column, bins=bins, **kwargs)

    # Add rug plot if requested
    if show_rug:
        _add_rug_plot(figure, data, column, backend, theme)

    return figure


def _add_rug_plot(
    figure: Any, data: "pd.DataFrame", column: str, backend: str, theme: TufteTheme
) -> None:
    """
    Add a rug plot showing individual data points.

    Args:
        figure: The figure object to modify.
        data: The pandas DataFrame containing the data.
        column: Column name for the data.
        backend: The backend being used.
        theme: The theme configuration.
    """
    if backend == "matplotlib":
        for ax in figure.axes:
            # Get y-axis limits to position rug at bottom
            ylim = ax.get_ylim()
            y_pos = ylim[0]

            # Add rug plot
            ax.plot(
                data[column],
                [y_pos] * len(data[column]),
                "|",
                color=theme.axis_color,
                alpha=0.5,
                markersize=8,
            )
    elif backend == "seaborn":
        # Seaborn has built-in rug plot support
        for ax in figure.axes:
            if HAS_SEABORN:
                sns.rugplot(
                    data=data, x=column, ax=ax, color=theme.axis_color, alpha=0.5
                )
    elif backend == "plotly":
        # Add rug plot as scatter trace at bottom
        figure.add_trace(
            go.Scatter(
                x=data[column],
                y=[0] * len(data[column]),
                mode="markers",
                marker=dict(
                    symbol="line-ns",
                    size=10,
                    color=theme.axis_color,
                    opacity=0.5,
                    line=dict(width=1),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
