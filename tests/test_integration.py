"""
Integration tests for TuftePlots library.
Tests end-to-end workflows and cross-component interactions.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os

from tufteplots import (
    tufte_line_plot,
    tufte_scatter_plot,
    tufte_bar_plot,
    tufte_histogram,
    small_multiples,
    apply_tufte_style,
    TufteTheme,
    ThemeManager,
)


class TestEndToEndWorkflows:
    """Test complete workflows from data to styled plot."""

    def test_complete_line_plot_workflow(self):
        """Test creating, styling, and exporting a line plot."""
        # Create data
        df = pd.DataFrame(
            {
                "x": range(10),
                "y": np.random.randn(10).cumsum(),
                "group": ["A"] * 5 + ["B"] * 5,
            }
        )

        # Create plot
        fig = tufte_line_plot(df, "x", "y", hue="group", backend="matplotlib")

        # Verify figure was created
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # Verify styling was applied
        ax = fig.axes[0]
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()

        plt.close(fig)

    def test_complete_scatter_plot_workflow(self):
        """Test creating scatter plot with trend line."""
        df = pd.DataFrame({"x": np.random.randn(50), "y": np.random.randn(50)})

        fig = tufte_scatter_plot(df, "x", "y", show_trend=True, backend="matplotlib")

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_complete_bar_plot_workflow(self):
        """Test creating bar plot with value labels."""
        df = pd.DataFrame({"category": ["A", "B", "C", "D"], "value": [10, 25, 15, 30]})

        fig = tufte_bar_plot(
            df, "category", "value", show_values=True, backend="matplotlib"
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_complete_histogram_workflow(self):
        """Test creating histogram with rug plot."""
        df = pd.DataFrame({"values": np.random.normal(0, 1, 100)})

        fig = tufte_histogram(df, "values", show_rug=True, backend="matplotlib")

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_small_multiples_workflow(self):
        """Test creating small multiples grid."""
        df = pd.DataFrame(
            {
                "x": list(range(10)) * 3,
                "y": np.random.randn(30).cumsum(),
                "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            }
        )

        fig = small_multiples(
            df, "x", "y", facet_by="group", plot_type="line", backend="matplotlib"
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3  # Three groups

        plt.close(fig)


class TestThemeCustomization:
    """Test theme customization workflows."""

    def test_custom_theme_application(self):
        """Test applying custom theme to plot."""
        custom_theme = TufteTheme(
            font_family="Inter",
            title_size=16,
            color_palette=["#FF0000", "#00FF00", "#0000FF"],
        )

        df = pd.DataFrame({"x": range(10), "y": np.random.randn(10)})

        fig = tufte_line_plot(df, "x", "y", backend="matplotlib", theme=custom_theme)

        assert fig is not None
        plt.close(fig)

    def test_theme_manager_workflow(self):
        """Test using ThemeManager for theme management."""
        manager = ThemeManager()

        # Update theme
        new_theme = manager.update_theme(title_size=18, show_grid=True)

        assert new_theme.title_size == 18
        assert new_theme.show_grid == True

        # Export and load theme
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            manager.export_theme(temp_path)
            loaded_theme = manager.load_theme(temp_path)

            assert loaded_theme.title_size == 18
            assert loaded_theme.show_grid == True
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestCrossBackendConsistency:
    """Test consistency across different backends."""

    def test_matplotlib_plotly_consistency(self):
        """Test that matplotlib and plotly produce similar outputs."""
        df = pd.DataFrame({"x": range(10), "y": np.random.randn(10)})

        # Create matplotlib plot
        fig_mpl = tufte_line_plot(df, "x", "y", backend="matplotlib")
        assert isinstance(fig_mpl, plt.Figure)
        plt.close(fig_mpl)

        # Create plotly plot
        fig_plotly = tufte_line_plot(df, "x", "y", backend="plotly")
        assert isinstance(fig_plotly, go.Figure)

    def test_apply_tufte_style_to_existing_plots(self):
        """Test applying Tufte style to pre-existing plots."""
        # Create matplotlib plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        # Apply Tufte style
        styled_fig = apply_tufte_style(fig, backend="matplotlib")

        assert styled_fig is fig  # Should modify in place
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()

        plt.close(fig)


class TestFileExport:
    """Test file export functionality."""

    def test_export_png(self):
        """Test exporting to PNG format."""
        df = pd.DataFrame({"x": range(10), "y": np.random.randn(10)})

        fig = tufte_line_plot(df, "x", "y", backend="matplotlib")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            fig.savefig(temp_path, dpi=150, bbox_inches="tight")
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            plt.close(fig)
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_pdf(self):
        """Test exporting to PDF format."""
        df = pd.DataFrame({"x": range(10), "y": np.random.randn(10)})

        fig = tufte_line_plot(df, "x", "y", backend="matplotlib")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = f.name

        try:
            fig.savefig(temp_path, bbox_inches="tight")
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            plt.close(fig)
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_svg(self):
        """Test exporting to SVG format."""
        df = pd.DataFrame({"x": range(10), "y": np.random.randn(10)})

        fig = tufte_line_plot(df, "x", "y", backend="matplotlib")

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            temp_path = f.name

        try:
            fig.savefig(temp_path, bbox_inches="tight")
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            plt.close(fig)
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises appropriate error."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})

        with pytest.raises(ValueError, match="Unsupported backend"):
            tufte_line_plot(df, "x", "y", backend="invalid_backend")

    def test_missing_columns_raises_error(self):
        """Test that missing columns raise appropriate error."""
        df = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(KeyError, match="Missing required columns"):
            tufte_line_plot(df, "x", "y", backend="matplotlib")

    def test_empty_dataframe_handled_gracefully(self):
        """Test that empty dataframes are handled gracefully."""
        df = pd.DataFrame({"x": [], "y": []})

        # Should not raise an error, but create an empty plot
        fig = tufte_line_plot(df, "x", "y", backend="matplotlib")
        assert fig is not None
        plt.close(fig)


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_large_dataset(self):
        """Test handling large datasets."""
        df = pd.DataFrame({"x": range(1000), "y": np.random.randn(1000).cumsum()})

        fig = tufte_line_plot(df, "x", "y", backend="matplotlib")
        assert fig is not None
        plt.close(fig)

    def test_multiple_groups_with_direct_labels(self):
        """Test multiple groups with direct labeling."""
        df = pd.DataFrame(
            {
                "x": list(range(20)) * 5,
                "y": np.random.randn(100).cumsum(),
                "group": ["Group A"] * 20
                + ["Group B"] * 20
                + ["Group C"] * 20
                + ["Group D"] * 20
                + ["Group E"] * 20,
            }
        )

        fig = tufte_line_plot(
            df, "x", "y", hue="group", direct_labels=True, backend="matplotlib"
        )
        assert fig is not None

        # Verify labels were added
        ax = fig.axes[0]
        texts = [child for child in ax.get_children() if hasattr(child, "get_text")]
        assert len(texts) > 0

        plt.close(fig)

    def test_datetime_x_axis(self):
        """Test handling datetime x-axis."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame({"date": dates, "value": np.random.randn(30).cumsum()})

        fig = tufte_line_plot(df, "date", "value", backend="matplotlib")
        assert fig is not None
        plt.close(fig)

    def test_categorical_data(self):
        """Test handling categorical data."""
        df = pd.DataFrame(
            {
                "category": ["Low", "Medium", "High", "Very High"],
                "value": [10, 25, 40, 30],
            }
        )

        fig = tufte_bar_plot(df, "category", "value", backend="matplotlib")
        assert fig is not None
        plt.close(fig)
