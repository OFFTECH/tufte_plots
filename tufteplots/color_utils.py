"""
Color conversion utilities for TuftePlots.

This module provides utilities to normalize color handling across different
plotting backends (matplotlib, plotly, seaborn), ensuring visual consistency
regardless of the backend used.
"""

from typing import Tuple, Union
import re


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., "#4e79a7" or "4e79a7").

    Returns:
        Tuple of (r, g, b) values in range 0-255.

    Raises:
        ValueError: If hex_color is not a valid hex color string.

    Example:
        >>> hex_to_rgb("#4e79a7")
        (78, 121, 167)
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip("#")

    # Validate hex color format
    if not re.match(r"^[0-9a-fA-F]{6}$", hex_color):
        raise ValueError(
            f"Invalid hex color '{hex_color}'. Expected format: #RRGGBB or RRGGBB"
        )

    # Convert to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (r, g, b)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to hex color string.

    Args:
        r: Red value (0-255).
        g: Green value (0-255).
        b: Blue value (0-255).

    Returns:
        Hex color string with '#' prefix (e.g., "#4e79a7").

    Raises:
        ValueError: If RGB values are out of range.

    Example:
        >>> rgb_to_hex(78, 121, 167)
        '#4e79a7'
    """
    # Validate RGB values
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError(f"RGB values must be in range 0-255. Got: r={r}, g={g}, b={b}")

    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb_normalized(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert hex color to normalized RGB tuple (0.0-1.0 range).

    This is useful for matplotlib which often uses normalized RGB values.

    Args:
        hex_color: Hex color string (e.g., "#4e79a7" or "4e79a7").

    Returns:
        Tuple of (r, g, b) values in range 0.0-1.0.

    Example:
        >>> hex_to_rgb_normalized("#4e79a7")
        (0.3058823529411765, 0.4745098039215686, 0.6549019607843137)
    """
    r, g, b = hex_to_rgb(hex_color)
    return (r / 255.0, g / 255.0, b / 255.0)


def rgb_normalized_to_hex(r: float, g: float, b: float) -> str:
    """
    Convert normalized RGB values (0.0-1.0) to hex color string.

    Args:
        r: Red value (0.0-1.0).
        g: Green value (0.0-1.0).
        b: Blue value (0.0-1.0).

    Returns:
        Hex color string with '#' prefix.

    Raises:
        ValueError: If RGB values are out of range.

    Example:
        >>> rgb_normalized_to_hex(0.306, 0.475, 0.655)
        '#4c7aa7'
    """
    # Validate normalized RGB values
    if not (0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError(
            f"Normalized RGB values must be in range 0.0-1.0. Got: r={r}, g={g}, b={b}"
        )

    # Convert to 0-255 range
    r_int = int(round(r * 255))
    g_int = int(round(g * 255))
    b_int = int(round(b * 255))

    return rgb_to_hex(r_int, g_int, b_int)


def normalize_color(color: Union[str, Tuple[float, float, float]]) -> str:
    """
    Normalize a color to hex format.

    This function accepts colors in various formats and converts them to
    a standard hex format for consistent handling across backends.

    Args:
        color: Color in hex format (str) or normalized RGB tuple (float, float, float).

    Returns:
        Hex color string with '#' prefix.

    Example:
        >>> normalize_color("#4e79a7")
        '#4e79a7'
        >>> normalize_color((0.306, 0.475, 0.655))
        '#4c7aa7'
    """
    if isinstance(color, str):
        # Already in hex format, just ensure it has '#' prefix
        if not color.startswith("#"):
            color = "#" + color
        # Validate and return
        hex_to_rgb(color)  # This will raise ValueError if invalid
        return color.lower()
    elif isinstance(color, tuple) and len(color) == 3:
        # Assume normalized RGB tuple
        return rgb_normalized_to_hex(color[0], color[1], color[2])
    else:
        raise ValueError(
            f"Unsupported color format: {color}. "
            "Expected hex string or RGB tuple (r, g, b)."
        )


def colors_match(
    color1: Union[str, Tuple[float, float, float]],
    color2: Union[str, Tuple[float, float, float]],
    tolerance: int = 2,
) -> bool:
    """
    Check if two colors match within a tolerance.

    This function compares two colors and returns True if they are equivalent
    within the specified tolerance. This is useful for testing cross-backend
    color consistency where minor rounding differences may occur.

    Args:
        color1: First color (hex string or normalized RGB tuple).
        color2: Second color (hex string or normalized RGB tuple).
        tolerance: Maximum allowed difference per RGB channel (0-255 scale).

    Returns:
        True if colors match within tolerance, False otherwise.

    Example:
        >>> colors_match("#4e79a7", "#4e79a7")
        True
        >>> colors_match("#4e79a7", "#4e79a9", tolerance=2)
        True
        >>> colors_match("#4e79a7", "#ff0000")
        False
    """
    # Normalize both colors to RGB tuples (0-255 scale)
    hex1 = normalize_color(color1)
    hex2 = normalize_color(color2)

    rgb1 = hex_to_rgb(hex1)
    rgb2 = hex_to_rgb(hex2)

    # Check if each channel is within tolerance
    for c1, c2 in zip(rgb1, rgb2):
        if abs(c1 - c2) > tolerance:
            return False

    return True


def palette_to_hex(palette: list) -> list:
    """
    Convert a color palette to hex format.

    This function normalizes all colors in a palette to hex format,
    ensuring consistent color representation across backends.

    Args:
        palette: List of colors in various formats (hex strings or RGB tuples).

    Returns:
        List of hex color strings.

    Example:
        >>> palette_to_hex(["#4e79a7", (0.95, 0.56, 0.17)])
        ['#4e79a7', '#f28f2b']
    """
    return [normalize_color(color) for color in palette]


def matplotlib_to_hex(color: Union[str, Tuple[float, float, float]]) -> str:
    """
    Convert matplotlib color format to hex.

    Matplotlib accepts colors in various formats including hex strings,
    normalized RGB tuples, and named colors. This function converts
    matplotlib colors to standard hex format.

    Args:
        color: Matplotlib color (hex string, normalized RGB tuple, or named color).

    Returns:
        Hex color string.

    Example:
        >>> matplotlib_to_hex((0.306, 0.475, 0.655))
        '#4c7aa7'
    """
    # For now, we handle hex and RGB tuples
    # Named colors would require matplotlib.colors.to_hex
    return normalize_color(color)


def plotly_to_hex(color: str) -> str:
    """
    Convert plotly color format to hex.

    Plotly primarily uses hex strings and RGB/RGBA strings.
    This function converts plotly colors to standard hex format.

    Args:
        color: Plotly color (hex string or rgb/rgba string).

    Returns:
        Hex color string.

    Example:
        >>> plotly_to_hex("#4e79a7")
        '#4e79a7'
        >>> plotly_to_hex("rgb(78, 121, 167)")
        '#4e79a7'
    """
    # Handle hex format
    if color.startswith("#"):
        return normalize_color(color)

    # Handle rgb() or rgba() format
    if color.startswith("rgb"):
        # Extract RGB values from string like "rgb(78, 121, 167)" or "rgba(78, 121, 167, 1.0)"
        match = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", color)
        if match:
            r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return rgb_to_hex(r, g, b)

    # If we can't parse it, try to normalize as-is
    return normalize_color(color)


def ensure_hex_palette(palette: list) -> list:
    """
    Ensure all colors in a palette are in hex format.

    This is a convenience function that converts a palette to hex format,
    handling various input formats from different backends.

    Args:
        palette: List of colors in various formats.

    Returns:
        List of hex color strings.

    Example:
        >>> ensure_hex_palette(["#4e79a7", "rgb(242, 142, 43)", (0.88, 0.34, 0.35)])
        ['#4e79a7', '#f28e2b', '#e05759']
    """
    result = []
    for color in palette:
        if isinstance(color, str):
            if color.startswith("rgb"):
                result.append(plotly_to_hex(color))
            else:
                result.append(normalize_color(color))
        elif isinstance(color, tuple):
            result.append(matplotlib_to_hex(color))
        else:
            raise ValueError(f"Unsupported color format: {color}")

    return result
