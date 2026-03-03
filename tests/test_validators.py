"""
Unit tests for ConfigValidator.

Tests validation functions for backends, colors, figures, and dataframes.
"""

import pytest
import pandas as pd
from tufteplots.validators import ConfigValidator


class TestValidateBackend:
    """Tests for validate_backend()."""

    def test_valid_backends(self):
        """Test that valid backends pass validation."""
        ConfigValidator.validate_backend("matplotlib")
        ConfigValidator.validate_backend("plotly")
        ConfigValidator.validate_backend("seaborn")

    def test_invalid_backend_raises_value_error(self):
        """Test that invalid backend raises ValueError with specific message."""
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate_backend("invalid_backend")

        error_msg = str(exc_info.value)
        assert "invalid_backend" in error_msg
        assert "Supported backends:" in error_msg
        assert "matplotlib" in error_msg
        assert "plotly" in error_msg
        assert "seaborn" in error_msg


class TestValidateColor:
    """Tests for validate_color()."""

    def test_valid_hex_colors(self):
        """Test that valid hex colors pass validation."""
        ConfigValidator.validate_color("#FF0000")
        ConfigValidator.validate_color("#00ff00")
        ConfigValidator.validate_color("#ABC")
        ConfigValidator.validate_color("#abc")

    def test_valid_named_colors(self):
        """Test that valid named colors pass validation."""
        ConfigValidator.validate_color("red")
        ConfigValidator.validate_color("blue")
        ConfigValidator.validate_color("white")
        ConfigValidator.validate_color("black")

    def test_invalid_hex_color_raises_value_error(self):
        """Test that invalid hex color raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate_color("#GGGGGG")

        error_msg = str(exc_info.value)
        assert "#GGGGGG" in error_msg
        assert "Expected hex (#RRGGBB) or named color" in error_msg

    def test_invalid_named_color_raises_value_error(self):
        """Test that invalid named color raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate_color("notacolor")

        error_msg = str(exc_info.value)
        assert "notacolor" in error_msg
        assert "Expected hex (#RRGGBB) or named color" in error_msg

    def test_non_string_color_raises_value_error(self):
        """Test that non-string color raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate_color(123)

        error_msg = str(exc_info.value)
        assert "Expected hex (#RRGGBB) or named color" in error_msg


class TestValidateFigure:
    """Tests for validate_figure()."""

    def test_matplotlib_figure_validation(self):
        """Test matplotlib figure validation."""
        try:
            import matplotlib.pyplot as plt

            fig, _ = plt.subplots()
            ConfigValidator.validate_figure(fig, "matplotlib")
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not properly configured")

    def test_wrong_figure_type_raises_type_error(self):
        """Test that wrong figure type raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ConfigValidator.validate_figure("not a figure", "matplotlib")

        error_msg = str(exc_info.value)
        assert "Expected matplotlib.figure.Figure" in error_msg
        assert "got" in error_msg


class TestValidateDataframeColumns:
    """Tests for validate_dataframe_columns()."""

    def test_valid_dataframe_with_required_columns(self):
        """Test that DataFrame with all required columns passes."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        ConfigValidator.validate_dataframe_columns(df, ["x", "y"])

    def test_missing_columns_raises_key_error(self):
        """Test that missing columns raises KeyError with specific message."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        with pytest.raises(KeyError) as exc_info:
            ConfigValidator.validate_dataframe_columns(df, ["x", "y", "z", "w"])

        error_msg = str(exc_info.value)
        assert "Missing required columns:" in error_msg
        assert "z" in error_msg
        assert "w" in error_msg

    def test_non_dataframe_raises_type_error(self):
        """Test that non-DataFrame raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ConfigValidator.validate_dataframe_columns([1, 2, 3], ["x"])

        error_msg = str(exc_info.value)
        assert "Expected pandas DataFrame" in error_msg
