"""
TuftePlots: A Python library for creating Tufte-style visualizations.

This library provides a unified API to create publication-ready, visually stunning
scientific plots across matplotlib, plotly, and seaborn following Edward Tufte's
principles of data visualization.

Key Principles
--------------
- **Maximize data-ink ratio**: Every pixel should convey information
- **Eliminate chartjunk**: Remove decorative elements that don't serve the data
- **Enable comparison**: Through small multiples and consistent scales
- **Direct communication**: Labels on data, not in legends

Quick Start
-----------
Create a simple Tufte-styled line plot:

    >>> import pandas as pd
    >>> import tufteplots as tp
    >>>
    >>> data = pd.DataFrame({'x': range(10), 'y': range(10)})
    >>> fig = tp.tufte_line_plot(data, 'x', 'y')
    >>> fig.savefig('plot.png')

Apply Tufte styling to an existing figure:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 2, 3])
    >>> fig = tp.apply_tufte_style(fig)

Create small multiples for comparison:

    >>> data = pd.DataFrame({
    ...     'x': list(range(10)) * 3,
    ...     'y': list(range(10)) * 3,
    ...     'category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10
    ... })
    >>> fig = tp.small_multiples(data, 'x', 'y', 'category')

Customize the theme:

    >>> theme = tp.TufteTheme(
    ...     font_family='Palatino',
    ...     title_size=16,
    ...     color_palette=['#4e79a7', '#f28e2b']
    ... )
    >>> fig = tp.tufte_line_plot(data, 'x', 'y', theme=theme)

Main Components
---------------
**High-level plotting functions:**
    - apply_tufte_style: Apply Tufte styling to existing figures
    - tufte_line_plot: Create Tufte-styled line plots
    - tufte_scatter_plot: Create Tufte-styled scatter plots
    - tufte_bar_plot: Create Tufte-styled bar plots
    - tufte_histogram: Create Tufte-styled histograms
    - small_multiples: Create grids of small multiple plots

**Configuration and theming:**
    - TufteTheme: Immutable theme configuration dataclass
    - PlotConfig: Per-plot configuration settings
    - ThemeManager: Manage and persist theme configurations

**Backend adapters:**
    - BackendAdapter: Abstract base class for backend adapters
    - MatplotlibAdapter: Adapter for matplotlib figures
    - PlotlyAdapter: Adapter for plotly figures
    - SeabornAdapter: Adapter for seaborn figures

**Utilities:**
    - LabelPositioner: Intelligent label positioning with collision avoidance
    - ConfigValidator: Input validation with clear error messages
    - Color utilities: Functions for color conversion and normalization

Supported Backends
------------------
- matplotlib: The default backend, widely used for static plots
- plotly: Interactive plots with hover tooltips and zoom
- seaborn: Statistical visualizations with Tufte styling

For more information, see the documentation at https://github.com/yourusername/tufteplots
"""

# Core theme and configuration
from tufteplots.theme import TufteTheme, PlotConfig, ThemeManager

# Backend adapters
from tufteplots.adapters import (
    BackendAdapter,
    MatplotlibAdapter,
    PlotlyAdapter,
    SeabornAdapter,
    small_multiples,
    calculate_grid_dimensions,
)

# Utilities
from tufteplots.label_positioner import LabelPositioner
from tufteplots.validators import ConfigValidator

# High-level API functions
from tufteplots.api import (
    apply_tufte_style,
    tufte_line_plot,
    tufte_scatter_plot,
    tufte_bar_plot,
    tufte_histogram,
)

# Color utilities
from tufteplots.color_utils import (
    normalize_color,
    colors_match,
    ensure_hex_palette,
    hex_to_rgb,
    rgb_to_hex,
    hex_to_rgb_normalized,
    rgb_normalized_to_hex,
)

__version__ = "0.1.0"
__author__ = "TuftePlots Contributors"
__license__ = "MIT"

__all__ = [
    # Theme and configuration
    "TufteTheme",
    "PlotConfig",
    "ThemeManager",
    # Backend adapters
    "BackendAdapter",
    "MatplotlibAdapter",
    "PlotlyAdapter",
    "SeabornAdapter",
    # High-level API
    "apply_tufte_style",
    "tufte_line_plot",
    "tufte_scatter_plot",
    "tufte_bar_plot",
    "tufte_histogram",
    "small_multiples",
    # Utilities
    "LabelPositioner",
    "ConfigValidator",
    "calculate_grid_dimensions",
    # Color utilities
    "normalize_color",
    "colors_match",
    "ensure_hex_palette",
    "hex_to_rgb",
    "rgb_to_hex",
    "hex_to_rgb_normalized",
    "rgb_normalized_to_hex",
]
