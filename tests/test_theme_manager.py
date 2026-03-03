"""
Unit tests for ThemeManager class.

This module contains unit tests for the ThemeManager class that manages
theme configuration and file I/O operations.
"""

import json
import os
import tempfile
import pytest
from tufteplots.theme import TufteTheme, ThemeManager


def test_theme_manager_init_default():
    """Test ThemeManager initialization with default theme."""
    manager = ThemeManager()
    theme = manager.get_theme()

    assert isinstance(theme, TufteTheme)
    assert theme.font_family == "Arial"
    assert theme.font_fallback == "sans-serif"
    assert theme.title_size == 18
    assert theme.label_size == 14


def test_theme_manager_init_custom():
    """Test ThemeManager initialization with custom theme."""
    custom_theme = TufteTheme(font_family="Arial", title_size=16, show_grid=True)
    manager = ThemeManager(theme=custom_theme)
    theme = manager.get_theme()

    assert theme.font_family == "Arial"
    assert theme.title_size == 16
    assert theme.show_grid is True


def test_theme_manager_get_theme():
    """Test getting the current theme."""
    manager = ThemeManager()
    theme1 = manager.get_theme()
    theme2 = manager.get_theme()

    # Should return the same theme instance
    assert theme1 is theme2


def test_theme_manager_update_theme_single_param():
    """Test updating a single theme parameter."""
    manager = ThemeManager()
    original_theme = manager.get_theme()

    new_theme = manager.update_theme(title_size=18)

    assert new_theme.title_size == 18
    # Other parameters should remain unchanged
    assert new_theme.font_family == original_theme.font_family
    assert new_theme.label_size == original_theme.label_size


def test_theme_manager_update_theme_multiple_params():
    """Test updating multiple theme parameters."""
    manager = ThemeManager()

    new_theme = manager.update_theme(
        title_size=20,
        font_family="Times New Roman",
        show_grid=True,
        color_palette=["#ff0000", "#00ff00", "#0000ff"],
    )

    assert new_theme.title_size == 20
    assert new_theme.font_family == "Times New Roman"
    assert new_theme.show_grid is True
    assert new_theme.color_palette == ["#ff0000", "#00ff00", "#0000ff"]


def test_theme_manager_update_theme_modifies_internal_state():
    """Test that update_theme modifies the manager's internal theme."""
    manager = ThemeManager()

    manager.update_theme(title_size=22)
    current_theme = manager.get_theme()

    assert current_theme.title_size == 22


def test_theme_manager_export_theme():
    """Test exporting theme to JSON file."""
    manager = ThemeManager()
    manager.update_theme(title_size=16, font_family="Arial")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        filepath = f.name

    try:
        manager.export_theme(filepath)

        # Verify file exists and contains valid JSON
        assert os.path.exists(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["title_size"] == 16
        assert data["font_family"] == "Arial"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_theme_manager_load_theme():
    """Test loading theme from JSON file."""
    # Create a custom theme and export it
    custom_theme = TufteTheme(
        font_family="Palatino",
        title_size=18,
        label_size=12,
        show_grid=True,
        color_palette=["#123456", "#abcdef"],
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        filepath = f.name
        f.write(custom_theme.to_json())

    try:
        # Load the theme with a new manager
        manager = ThemeManager()
        loaded_theme = manager.load_theme(filepath)

        assert loaded_theme.font_family == "Palatino"
        assert loaded_theme.title_size == 18
        assert loaded_theme.label_size == 12
        assert loaded_theme.show_grid is True
        assert loaded_theme.color_palette == ["#123456", "#abcdef"]

        # Verify the manager's internal theme was updated
        assert manager.get_theme() == loaded_theme
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_theme_manager_load_theme_file_not_found():
    """Test loading theme from non-existent file raises FileNotFoundError."""
    manager = ThemeManager()

    with pytest.raises(FileNotFoundError):
        manager.load_theme("nonexistent_file.json")


def test_theme_manager_load_theme_invalid_json():
    """Test loading theme from invalid JSON raises JSONDecodeError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        filepath = f.name
        f.write("{ invalid json content }")

    try:
        manager = ThemeManager()

        with pytest.raises(json.JSONDecodeError):
            manager.load_theme(filepath)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_theme_manager_export_load_roundtrip():
    """Test that exporting and loading preserves theme configuration."""
    original_manager = ThemeManager()
    original_manager.update_theme(
        font_family="Garamond",
        title_size=15,
        label_size=10,
        tick_size=8,
        line_width=2.0,
        show_grid=True,
        use_range_frame=False,
        color_palette=["#111111", "#222222", "#333333"],
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        filepath = f.name

    try:
        # Export
        original_manager.export_theme(filepath)

        # Load with new manager
        new_manager = ThemeManager()
        loaded_theme = new_manager.load_theme(filepath)

        # Verify all properties match
        original_theme = original_manager.get_theme()
        assert loaded_theme == original_theme
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
