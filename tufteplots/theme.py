"""
Theme configuration for TuftePlots.

This module contains the TufteTheme and PlotConfig dataclasses that define
the visual styling parameters for Tufte-style visualizations.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import json


DEFAULT_COLOR_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
]


@dataclass
class TufteTheme:
    """
    Immutable theme configuration for Tufte-style plots.

    This dataclass contains all styling parameters that define the visual
    appearance of plots following Edward Tufte's design principles.
    """

    # Typography
    font_family: str = "Arial"
    font_fallback: str = "sans-serif"
    title_size: int = 18
    label_size: int = 14
    tick_size: int = 12

    # Lines and borders
    line_width: float = 1.5
    axis_line_width: float = 0.5

    # Colors
    background_color: str = "white"
    axis_color: str = "#333333"
    grid_color: str = "#eeeeee"
    text_color: str = "#333333"
    color_palette: List[str] = field(
        default_factory=lambda: DEFAULT_COLOR_PALETTE.copy()
    )

    # Tufte features
    show_grid: bool = False
    use_range_frame: bool = True
    remove_legend: bool = True
    direct_labels: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert theme to dictionary for serialization.

        Returns:
            Dictionary containing all theme parameters.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TufteTheme":
        """
        Create theme from dictionary.

        Args:
            data: Dictionary containing theme parameters.

        Returns:
            TufteTheme instance with the specified parameters.
        """
        return cls(**data)

    def to_json(self) -> str:
        """
        Serialize theme to JSON string.

        Returns:
            JSON string representation of the theme.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TufteTheme":
        """
        Deserialize theme from JSON string.

        Args:
            json_str: JSON string containing theme parameters.

        Returns:
            TufteTheme instance with the specified parameters.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class PlotConfig:
    """
    Per-plot configuration settings.

    This dataclass contains settings that can be customized for individual
    plots, such as titles, labels, and figure dimensions.
    """

    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    figsize: Tuple[float, float] = (8.0, 5.0)
    dpi: int = 150
    labels: Optional[List[str]] = None
    label_positions: Optional[List[Tuple[float, float]]] = None


class ThemeManager:
    """
    Manages Tufte theme configuration and provides file I/O operations.

    This class handles theme creation, customization, and persistence,
    allowing users to save and load theme configurations from JSON files.
    """

    def __init__(self, theme: Optional[TufteTheme] = None):
        """
        Initialize ThemeManager with a theme.

        Args:
            theme: Optional TufteTheme instance. If None, uses default theme.
        """
        self._theme = theme if theme is not None else TufteTheme()

    def get_theme(self) -> TufteTheme:
        """
        Return the current theme configuration.

        Returns:
            The current TufteTheme instance.
        """
        return self._theme

    def update_theme(self, **kwargs) -> TufteTheme:
        """
        Return a new theme with updated parameters.

        This method creates a new TufteTheme instance with the specified
        parameters updated, leaving the original theme unchanged.

        Args:
            **kwargs: Theme parameters to update (e.g., font_family="Arial",
                     title_size=16, color_palette=["#ff0000", "#00ff00"]).

        Returns:
            A new TufteTheme instance with updated parameters.

        Example:
            >>> manager = ThemeManager()
            >>> new_theme = manager.update_theme(title_size=16, show_grid=True)
            >>> manager._theme = new_theme  # Update the manager's theme
        """
        # Get current theme as dict
        current_dict = self._theme.to_dict()

        # Update with new values
        current_dict.update(kwargs)

        # Create and return new theme
        new_theme = TufteTheme.from_dict(current_dict)
        self._theme = new_theme
        return new_theme

    def export_theme(self, filepath: str) -> None:
        """
        Serialize the current theme to a JSON file.

        Args:
            filepath: Path where the JSON file should be saved.

        Raises:
            IOError: If the file cannot be written.

        Example:
            >>> manager = ThemeManager()
            >>> manager.export_theme("my_theme.json")
        """
        json_str = self._theme.to_json()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_str)

    def load_theme(self, filepath: str) -> TufteTheme:
        """
        Load a theme from a JSON file.

        Args:
            filepath: Path to the JSON file containing theme configuration.

        Returns:
            The loaded TufteTheme instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.

        Example:
            >>> manager = ThemeManager()
            >>> theme = manager.load_theme("my_theme.json")
        """
        with open(filepath, "r", encoding="utf-8") as f:
            json_str = f.read()

        loaded_theme = TufteTheme.from_json(json_str)
        self._theme = loaded_theme
        return loaded_theme
