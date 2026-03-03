"""
Input validation for TuftePlots.

This module contains the ConfigValidator class that validates user input
and provides clear error messages for invalid configurations.
"""

import re
from typing import Any, List, Set

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


SUPPORTED_BACKENDS: Set[str] = {"matplotlib", "plotly", "seaborn"}

# Named colors that are commonly supported
NAMED_COLORS: Set[str] = {
    "white",
    "black",
    "red",
    "green",
    "blue",
    "yellow",
    "cyan",
    "magenta",
    "gray",
    "grey",
    "orange",
    "purple",
    "pink",
    "brown",
    "navy",
    "teal",
    "olive",
    "maroon",
    "aqua",
    "lime",
    "silver",
    "fuchsia",
}

# Regex pattern for hex colors
HEX_COLOR_PATTERN = re.compile(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$")


class ConfigValidator:
    """
    Validates user input and provides clear error messages.

    This class contains static methods for validating various types of
    configuration inputs used throughout the TuftePlots library.
    """

    SUPPORTED_BACKENDS = SUPPORTED_BACKENDS

    @staticmethod
    def validate_backend(backend: str) -> None:
        """
        Validate that the backend is supported.

        Args:
            backend: The backend name to validate.

        Raises:
            ValueError: If the backend is not supported.
        """
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                f"Supported backends: {SUPPORTED_BACKENDS}"
            )

    @staticmethod
    def validate_color(color: str) -> None:
        """
        Validate that the color format is valid.

        Args:
            color: The color value to validate (hex or named color).

        Raises:
            ValueError: If the color format is invalid.
        """
        if not isinstance(color, str):
            raise ValueError(
                f"Invalid color '{color}'. Expected hex (#RRGGBB) or named color"
            )

        color_lower = color.lower()

        # Check if it's a valid hex color
        if color.startswith("#"):
            if not HEX_COLOR_PATTERN.match(color):
                raise ValueError(
                    f"Invalid color '{color}'. Expected hex (#RRGGBB) or named color"
                )
            return

        # Check if it's a named color
        if color_lower not in NAMED_COLORS:
            raise ValueError(
                f"Invalid color '{color}'. Expected hex (#RRGGBB) or named color"
            )

    @staticmethod
    def validate_figure(figure: Any, backend: str) -> None:
        """
        Validate that the figure type matches the expected backend.

        Args:
            figure: The figure object to validate.
            backend: The expected backend name.

        Raises:
            TypeError: If the figure type doesn't match the backend.
        """
        figure_type = type(figure).__module__ + "." + type(figure).__name__

        if backend == "matplotlib":
            if "matplotlib.figure.Figure" not in figure_type:
                raise TypeError(f"Expected matplotlib.figure.Figure, got {figure_type}")
        elif backend == "plotly":
            if (
                "plotly" not in figure_type.lower()
                or "figure" not in figure_type.lower()
            ):
                raise TypeError(
                    f"Expected plotly.graph_objs._figure.Figure, got {figure_type}"
                )
        elif backend == "seaborn":
            valid = (
                "matplotlib.figure.Figure" in figure_type
                or "seaborn" in figure_type.lower()
            )
            if not valid:
                raise TypeError(
                    f"Expected matplotlib.figure.Figure or seaborn.axisgrid.FacetGrid, "
                    f"got {figure_type}"
                )

    @staticmethod
    def validate_dataframe_columns(df: Any, required: List[str]) -> None:
        """
        Validate that required columns exist in the DataFrame.

        Args:
            df: The pandas DataFrame to validate.
            required: List of required column names.

        Raises:
            KeyError: If any required columns are missing.
        """
        if not HAS_PANDAS:
            return

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
