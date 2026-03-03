"""
Stress tests for TuftePlots library.
Tests performance and stability under extreme conditions.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tufteplots import (
    tufte_line_plot,
    tufte_scatter_plot,
    tufte_bar_plot,
    small_multiples,
)


class TestPerformance:
    """Test performance with large datasets."""

    def test_large_line_plot(self):
        """Test line plot with 10,000 points."""
        df = pd.DataFrame({"x": range(10000), "y": np.random.randn(10000).cumsum()})

        fig = tufte_line_plot(df, "x", "y", backend="matplotlib")
        assert fig is not None
        plt.close(fig)

    def test_many_series(self):
        """Test line plot with 20 series."""
        n_points = 100
        n_series = 20

        df = pd.DataFrame(
            {
                "x": list(range(n_points)) * n_series,
                "y": np.random.randn(n_points * n_series).cumsum(),
                "group": [
                    f"Series {i}" for i in range(n_series) for _ in range(n_points)
                ],
            }
        )

        fig = tufte_line_plot(df, "x", "y", hue="group", backend="matplotlib")
        assert fig is not None
        plt.close(fig)

    def test_large_scatter(self):
        """Test scatter plot with 5,000 points."""
        df = pd.DataFrame({"x": np.random.randn(5000), "y": np.random.randn(5000)})

        fig = tufte_scatter_plot(df, "x", "y", backend="matplotlib")
        assert fig is not None
        plt.close(fig)

    def test_many_bars(self):
        """Test bar plot with 100 categories."""
        df = pd.DataFrame(
            {
                "category": [f"Cat{i}" for i in range(100)],
                "value": np.random.randint(1, 100, 100),
            }
        )

        fig = tufte_bar_plot(df, "category", "value", backend="matplotlib")
        assert fig is not None
        plt.close(fig)

    def test_many_small_multiples(self):
        """Test small multiples with 12 groups."""
        n_points = 50
        n_groups = 12

        df = pd.DataFrame(
            {
                "x": list(range(n_points)) * n_groups,
                "y": np.random.randn(n_points * n_groups).cumsum(),
                "group": [
                    f"Group {i}" for i in range(n_groups) for _ in range(n_points)
                ],
            }
        )

        fig = small_multiples(df, "x", "y", facet_by="group", backend="matplotlib")
        assert fig is not None
        assert len(fig.axes) == n_groups
        plt.close(fig)


class TestExtremeValues:
    """Test handling of extreme numerical values."""

    def test_very_large_numbers(self):
        """Test with numbers near float max."""
        df = pd.DataFrame(
            {
                "x": range(10),
                "y": np.array(
                    [
                        1e100,
                        2e100,
                        3e100,
                        4e100,
                        5e100,
                        6e100,
                        7e100,
                        8e100,
                        9e100,
                        1e101,
                    ]
                ),
            }
        )

        fig = tufte_line_plot(df, "x", "y", backend="matplotlib")
        assert fig is not None
        plt.close(fig)

    def test_very_small_numbers(self):
        """Test with numbers near zero."""
        df = pd.DataFrame(
            {
                "x": range(10),
                "y": np.array(
                    [
                        1e-100,
                        2e-100,
                        3e-100,
                        4e-100,
                        5e-100,
                        6e-100,
                        7e-100,
                        8e-100,
                        9e-100,
                        1e-99,
                    ]
                ),
            }
        )

        fig = tufte_line_plot(df, "x", "y", backend="matplotlib")
        assert fig is not None
        plt.close(fig)

    def test_mixed_scales(self):
        """Test with values spanning many orders of magnitude."""
        df = pd.DataFrame(
            {
                "x": range(10),
                "y": [1e-10, 1e-5, 1e-2, 1, 10, 100, 1e5, 1e10, 1e15, 1e20],
            }
        )

        fig = tufte_line_plot(df, "x", "y", backend="matplotlib")
        assert fig is not None
        plt.close(fig)

    def test_all_zeros(self):
        """Test with all zero values."""
        df = pd.DataFrame({"x": range(10), "y": [0] * 10})

        fig = tufte_line_plot(df, "x", "y", backend="matplotlib")
        assert fig is not None
        plt.close(fig)

    def test_alternating_signs(self):
        """Test with rapidly alternating positive/negative values."""
        df = pd.DataFrame(
            {"x": range(100), "y": [(-1) ** i * 1000 for i in range(100)]}
        )

        fig = tufte_line_plot(df, "x", "y", backend="matplotlib")
        assert fig is not None
        plt.close(fig)


class TestMemoryStress:
    """Test memory handling with repeated operations."""

    def test_repeated_plot_creation(self):
        """Test creating and closing many plots."""
        df = pd.DataFrame({"x": range(100), "y": np.random.randn(100)})

        # Create and close 50 plots
        for _ in range(50):
            fig = tufte_line_plot(df, "x", "y", backend="matplotlib")
            plt.close(fig)

        # Should complete without memory issues
        assert True

    def test_repeated_style_application(self):
        """Test applying style repeatedly to same figure."""
        from tufteplots import apply_tufte_style

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        # Apply style 20 times
        for _ in range(20):
            apply_tufte_style(fig, backend="matplotlib")

        plt.close(fig)
        assert True


class TestConcurrentOperations:
    """Test handling of multiple operations."""

    def test_multiple_backends_same_data(self):
        """Test creating plots with different backends for same data."""
        df = pd.DataFrame({"x": range(50), "y": np.random.randn(50)})

        # Create with matplotlib
        fig_mpl = tufte_line_plot(df, "x", "y", backend="matplotlib")
        assert fig_mpl is not None

        # Create with plotly
        fig_plotly = tufte_line_plot(df, "x", "y", backend="plotly")
        assert fig_plotly is not None

        # Create with seaborn
        fig_sns = tufte_line_plot(df, "x", "y", backend="seaborn")
        assert fig_sns is not None

        plt.close("all")

    def test_multiple_plot_types_same_data(self):
        """Test creating different plot types from same data."""
        df = pd.DataFrame(
            {
                "x": range(50),
                "y": np.random.randn(50).cumsum(),
                "category": ["A", "B"] * 25,
            }
        )

        fig1 = tufte_line_plot(df, "x", "y", backend="matplotlib")
        fig2 = tufte_scatter_plot(df, "x", "y", backend="matplotlib")

        # For bar plot, need aggregated data
        df_agg = df.groupby("category")["y"].mean().reset_index()
        fig3 = tufte_bar_plot(df_agg, "category", "y", backend="matplotlib")

        assert all([fig1, fig2, fig3])
        plt.close("all")


class TestDataIntegrity:
    """Test that operations don't modify input data."""

    def test_dataframe_not_modified(self):
        """Test that creating plot doesn't modify input DataFrame."""
        df = pd.DataFrame({"x": range(10), "y": np.random.randn(10)})

        df_copy = df.copy()

        fig = tufte_line_plot(df, "x", "y", backend="matplotlib")

        # DataFrame should be unchanged
        pd.testing.assert_frame_equal(df, df_copy)

        plt.close(fig)

    def test_theme_not_modified(self):
        """Test that using theme doesn't modify original theme object."""
        from tufteplots import TufteTheme

        theme = TufteTheme(title_size=16)
        original_size = theme.title_size

        df = pd.DataFrame({"x": range(10), "y": np.random.randn(10)})

        fig = tufte_line_plot(df, "x", "y", theme=theme, backend="matplotlib")

        # Theme should be unchanged
        assert theme.title_size == original_size

        plt.close(fig)
