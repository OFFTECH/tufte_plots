"""
Unit tests for edge cases.

Tests edge cases including empty DataFrames, single data points,
extreme values, and font fallback behavior.

Requirements: 3.4, 9.1, 9.2, 9.3, 9.4
"""

import pytest
import pandas as pd
import numpy as np
import warnings
import logging

try:
    import matplotlib.pyplot as plt
    import matplotlib.figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from tufteplots.api import (
    tufte_line_plot,
    tufte_scatter_plot,
    tufte_bar_plot,
    tufte_histogram,
    apply_tufte_style,
)
from tufteplots.theme import TufteTheme
from tufteplots.validators import ConfigValidator


class TestEmptyDataFrames:
    """Tests for handling empty DataFrames."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_line_plot_empty_dataframe(self):
        """Test that line plot handles empty DataFrame gracefully."""
        data = pd.DataFrame({"x": [], "y": []})

        # Should create a figure without crashing
        fig = tufte_line_plot(data, "x", "y", direct_labels=False)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_scatter_plot_empty_dataframe(self):
        """Test that scatter plot handles empty DataFrame gracefully."""
        data = pd.DataFrame({"x": [], "y": []})

        # Should create a figure without crashing
        fig = tufte_scatter_plot(data, "x", "y")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_bar_plot_empty_dataframe(self):
        """Test that bar plot handles empty DataFrame gracefully."""
        data = pd.DataFrame({"category": [], "value": []})

        # Should create a figure without crashing
        fig = tufte_bar_plot(data, "category", "value", show_values=False)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_histogram_empty_dataframe(self):
        """Test that histogram handles empty DataFrame gracefully."""
        data = pd.DataFrame({"values": []})

        # Should create a figure without crashing
        fig = tufte_histogram(data, "values", show_rug=False)
        assert fig is not None
        plt.close(fig)


class TestSingleDataPoint:
    """Tests for handling single data points."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_line_plot_single_point(self):
        """Test that line plot handles single data point."""
        data = pd.DataFrame({"x": [1], "y": [1]})

        fig = tufte_line_plot(data, "x", "y", direct_labels=False)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_scatter_plot_single_point(self):
        """Test that scatter plot handles single data point."""
        data = pd.DataFrame({"x": [1], "y": [1]})

        fig = tufte_scatter_plot(data, "x", "y")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_scatter_plot_single_point_with_trend(self):
        """Test that scatter plot with trend line handles single point."""
        data = pd.DataFrame({"x": [1], "y": [1]})

        # Should not crash even though trend line can't be calculated
        fig = tufte_scatter_plot(data, "x", "y", show_trend=True)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_bar_plot_single_bar(self):
        """Test that bar plot handles single bar."""
        data = pd.DataFrame({"category": ["A"], "value": [10]})

        fig = tufte_bar_plot(data, "category", "value")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_histogram_single_value(self):
        """Test that histogram handles single value."""
        data = pd.DataFrame({"values": [5.0]})

        fig = tufte_histogram(data, "values", show_rug=False)
        assert fig is not None
        plt.close(fig)


class TestExtremeValues:
    """Tests for handling extreme values."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_line_plot_very_large_values(self):
        """Test that line plot handles very large values."""
        data = pd.DataFrame({"x": [1, 2, 3], "y": [1e10, 2e10, 3e10]})

        fig = tufte_line_plot(data, "x", "y", direct_labels=False)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_line_plot_very_small_values(self):
        """Test that line plot handles very small values."""
        data = pd.DataFrame({"x": [1, 2, 3], "y": [1e-10, 2e-10, 3e-10]})

        fig = tufte_line_plot(data, "x", "y", direct_labels=False)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_scatter_plot_negative_values(self):
        """Test that scatter plot handles negative values."""
        data = pd.DataFrame({"x": [-100, -50, -10], "y": [-200, -100, -50]})

        fig = tufte_scatter_plot(data, "x", "y")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_bar_plot_negative_values(self):
        """Test that bar plot handles negative values."""
        data = pd.DataFrame({"category": ["A", "B", "C"], "value": [-10, -20, -5]})

        fig = tufte_bar_plot(data, "category", "value")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_histogram_with_outliers(self):
        """Test that histogram handles data with extreme outliers."""
        # Most values around 0, but with extreme outliers
        values = [0, 1, 2, 3, 4, 5, 1000, -1000]
        data = pd.DataFrame({"values": values})

        fig = tufte_histogram(data, "values", show_rug=False)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_line_plot_with_nan_values(self):
        """Test that line plot handles NaN values."""
        data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, np.nan, 3, np.nan, 5]})

        # Should handle NaN values without crashing
        fig = tufte_line_plot(data, "x", "y", direct_labels=False)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_scatter_plot_with_inf_values(self):
        """Test that scatter plot handles infinite values."""
        data = pd.DataFrame({"x": [1, 2, 3], "y": [1, np.inf, 3]})

        # Should handle inf values without crashing
        fig = tufte_scatter_plot(data, "x", "y")
        assert fig is not None
        plt.close(fig)


class TestFontFallback:
    """Tests for font fallback behavior."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_nonexistent_font_falls_back(self):
        """Test that nonexistent font falls back to system font."""
        # Create theme with nonexistent font
        theme = TufteTheme(
            font_family="NonexistentFontThatDoesNotExist123",
            font_fallback="sans-serif",
        )

        data = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})

        # Should not crash, should fall back to system font
        # We can't easily test if a warning was logged, but we can verify it doesn't crash
        fig = tufte_line_plot(data, "x", "y", theme=theme, direct_labels=False)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_apply_tufte_style_with_nonexistent_font(self):
        """Test that apply_tufte_style handles nonexistent font."""
        theme = TufteTheme(
            font_family="AnotherNonexistentFont999", font_fallback="serif"
        )

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Should not crash
        styled_fig = apply_tufte_style(fig, theme=theme)
        assert styled_fig is not None
        plt.close(fig)


class TestValidationEdgeCases:
    """Tests for validation edge cases."""

    def test_validate_backend_case_sensitivity(self):
        """Test that backend validation is case-sensitive."""
        # Lowercase should work
        ConfigValidator.validate_backend("matplotlib")

        # Uppercase should fail
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate_backend("MATPLOTLIB")

        assert "MATPLOTLIB" in str(exc_info.value)

    def test_validate_color_empty_string(self):
        """Test that empty string color raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate_color("")

        assert "Expected hex (#RRGGBB) or named color" in str(exc_info.value)

    def test_validate_dataframe_empty_required_list(self):
        """Test validation with empty required columns list."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # Should pass with no required columns
        ConfigValidator.validate_dataframe_columns(df, [])

    def test_validate_dataframe_none_value(self):
        """Test that None instead of DataFrame raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ConfigValidator.validate_dataframe_columns(None, ["x"])

        assert "Expected pandas DataFrame" in str(exc_info.value)


class TestMultiSeriesEdgeCases:
    """Tests for multi-series edge cases."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_line_plot_single_group_in_hue(self):
        """Test line plot with hue column containing only one unique value."""
        data = pd.DataFrame(
            {"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5], "category": ["A"] * 5}
        )

        fig = tufte_line_plot(data, "x", "y", hue="category")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_line_plot_many_groups(self):
        """Test line plot with many groups (stress test)."""
        # Create data with 20 different groups
        n_groups = 20
        n_points = 10

        data_list = []
        for i in range(n_groups):
            for j in range(n_points):
                data_list.append({"x": j, "y": j + i, "category": f"Group_{i}"})

        data = pd.DataFrame(data_list)

        fig = tufte_line_plot(data, "x", "y", hue="category", direct_labels=False)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_scatter_plot_with_all_same_values(self):
        """Test scatter plot where all points have the same coordinates."""
        data = pd.DataFrame({"x": [5, 5, 5, 5, 5], "y": [10, 10, 10, 10, 10]})

        fig = tufte_scatter_plot(data, "x", "y")
        assert fig is not None
        plt.close(fig)


class TestDataTypeEdgeCases:
    """Tests for different data types."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_bar_plot_with_string_categories(self):
        """Test bar plot with long string categories."""
        data = pd.DataFrame(
            {
                "category": [
                    "Very Long Category Name A",
                    "Very Long Category Name B",
                    "Very Long Category Name C",
                ],
                "value": [10, 20, 15],
            }
        )

        fig = tufte_bar_plot(data, "category", "value")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_line_plot_with_datetime_x_axis(self):
        """Test line plot with datetime x-axis."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({"date": dates, "value": range(10)})

        # Should handle datetime without crashing
        fig = tufte_line_plot(data, "date", "value", direct_labels=False)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_histogram_with_integer_data(self):
        """Test histogram with integer data."""
        data = pd.DataFrame({"values": [1, 2, 2, 3, 3, 3, 4, 4, 5]})

        fig = tufte_histogram(data, "values", show_rug=False)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_histogram_with_all_same_value(self):
        """Test histogram where all values are identical."""
        data = pd.DataFrame({"values": [5.0] * 100})

        fig = tufte_histogram(data, "values", show_rug=False)
        assert fig is not None
        plt.close(fig)
