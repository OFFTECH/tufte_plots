"""
Tests for color utility functions.

This module tests the color conversion and normalization utilities
that ensure consistent color handling across different backends.
"""

import pytest
from tufteplots.color_utils import (
    hex_to_rgb,
    rgb_to_hex,
    hex_to_rgb_normalized,
    rgb_normalized_to_hex,
    normalize_color,
    colors_match,
    palette_to_hex,
    matplotlib_to_hex,
    plotly_to_hex,
    ensure_hex_palette,
)


class TestHexToRgb:
    """Tests for hex_to_rgb function."""

    def test_hex_to_rgb_with_hash(self):
        """Test conversion of hex color with # prefix."""
        assert hex_to_rgb("#4e79a7") == (78, 121, 167)

    def test_hex_to_rgb_without_hash(self):
        """Test conversion of hex color without # prefix."""
        assert hex_to_rgb("4e79a7") == (78, 121, 167)

    def test_hex_to_rgb_uppercase(self):
        """Test conversion of uppercase hex color."""
        assert hex_to_rgb("#4E79A7") == (78, 121, 167)

    def test_hex_to_rgb_invalid_format(self):
        """Test that invalid hex format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid hex color"):
            hex_to_rgb("#zzz")

    def test_hex_to_rgb_wrong_length(self):
        """Test that wrong length hex string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid hex color"):
            hex_to_rgb("#fff")


class TestRgbToHex:
    """Tests for rgb_to_hex function."""

    def test_rgb_to_hex_basic(self):
        """Test conversion of RGB values to hex."""
        assert rgb_to_hex(78, 121, 167) == "#4e79a7"

    def test_rgb_to_hex_black(self):
        """Test conversion of black color."""
        assert rgb_to_hex(0, 0, 0) == "#000000"

    def test_rgb_to_hex_white(self):
        """Test conversion of white color."""
        assert rgb_to_hex(255, 255, 255) == "#ffffff"

    def test_rgb_to_hex_out_of_range(self):
        """Test that out of range RGB values raise ValueError."""
        with pytest.raises(ValueError, match="RGB values must be in range 0-255"):
            rgb_to_hex(256, 0, 0)

        with pytest.raises(ValueError, match="RGB values must be in range 0-255"):
            rgb_to_hex(0, -1, 0)


class TestHexToRgbNormalized:
    """Tests for hex_to_rgb_normalized function."""

    def test_hex_to_rgb_normalized_basic(self):
        """Test conversion to normalized RGB."""
        r, g, b = hex_to_rgb_normalized("#4e79a7")
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0
        # Check approximate values
        assert abs(r - 78 / 255.0) < 0.001
        assert abs(g - 121 / 255.0) < 0.001
        assert abs(b - 167 / 255.0) < 0.001

    def test_hex_to_rgb_normalized_black(self):
        """Test conversion of black to normalized RGB."""
        assert hex_to_rgb_normalized("#000000") == (0.0, 0.0, 0.0)

    def test_hex_to_rgb_normalized_white(self):
        """Test conversion of white to normalized RGB."""
        r, g, b = hex_to_rgb_normalized("#ffffff")
        assert abs(r - 1.0) < 0.001
        assert abs(g - 1.0) < 0.001
        assert abs(b - 1.0) < 0.001


class TestRgbNormalizedToHex:
    """Tests for rgb_normalized_to_hex function."""

    def test_rgb_normalized_to_hex_basic(self):
        """Test conversion from normalized RGB to hex."""
        result = rgb_normalized_to_hex(0.306, 0.475, 0.655)
        # Should be close to #4e79a7
        assert result.startswith("#")
        assert len(result) == 7

    def test_rgb_normalized_to_hex_black(self):
        """Test conversion of black from normalized RGB."""
        assert rgb_normalized_to_hex(0.0, 0.0, 0.0) == "#000000"

    def test_rgb_normalized_to_hex_white(self):
        """Test conversion of white from normalized RGB."""
        assert rgb_normalized_to_hex(1.0, 1.0, 1.0) == "#ffffff"

    def test_rgb_normalized_to_hex_out_of_range(self):
        """Test that out of range normalized values raise ValueError."""
        with pytest.raises(
            ValueError, match="Normalized RGB values must be in range 0.0-1.0"
        ):
            rgb_normalized_to_hex(1.5, 0.5, 0.5)


class TestNormalizeColor:
    """Tests for normalize_color function."""

    def test_normalize_color_hex_with_hash(self):
        """Test normalization of hex color with #."""
        assert normalize_color("#4e79a7") == "#4e79a7"

    def test_normalize_color_hex_without_hash(self):
        """Test normalization of hex color without #."""
        assert normalize_color("4e79a7") == "#4e79a7"

    def test_normalize_color_uppercase(self):
        """Test normalization converts to lowercase."""
        assert normalize_color("#4E79A7") == "#4e79a7"

    def test_normalize_color_rgb_tuple(self):
        """Test normalization of RGB tuple."""
        result = normalize_color((0.306, 0.475, 0.655))
        assert result.startswith("#")
        assert len(result) == 7

    def test_normalize_color_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported color format"):
            normalize_color([1, 2, 3])


class TestColorsMatch:
    """Tests for colors_match function."""

    def test_colors_match_identical_hex(self):
        """Test that identical hex colors match."""
        assert colors_match("#4e79a7", "#4e79a7")

    def test_colors_match_within_tolerance(self):
        """Test that colors within tolerance match."""
        assert colors_match("#4e79a7", "#4e79a9", tolerance=2)

    def test_colors_match_outside_tolerance(self):
        """Test that colors outside tolerance don't match."""
        assert not colors_match("#4e79a7", "#ff0000", tolerance=2)

    def test_colors_match_mixed_formats(self):
        """Test matching with mixed color formats."""
        # Convert hex to normalized RGB and compare
        hex_color = "#4e79a7"
        r, g, b = hex_to_rgb_normalized(hex_color)
        assert colors_match(hex_color, (r, g, b), tolerance=2)


class TestPaletteToHex:
    """Tests for palette_to_hex function."""

    def test_palette_to_hex_all_hex(self):
        """Test conversion of palette with all hex colors."""
        palette = ["#4e79a7", "#f28e2b", "#e15759"]
        result = palette_to_hex(palette)
        assert result == ["#4e79a7", "#f28e2b", "#e15759"]

    def test_palette_to_hex_mixed_formats(self):
        """Test conversion of palette with mixed formats."""
        palette = ["#4e79a7", (0.95, 0.56, 0.17)]
        result = palette_to_hex(palette)
        assert len(result) == 2
        assert result[0] == "#4e79a7"
        assert result[1].startswith("#")


class TestMatplotlibToHex:
    """Tests for matplotlib_to_hex function."""

    def test_matplotlib_to_hex_rgb_tuple(self):
        """Test conversion of matplotlib RGB tuple."""
        result = matplotlib_to_hex((0.306, 0.475, 0.655))
        assert result.startswith("#")
        assert len(result) == 7

    def test_matplotlib_to_hex_hex_string(self):
        """Test conversion of matplotlib hex string."""
        assert matplotlib_to_hex("#4e79a7") == "#4e79a7"


class TestPlotlyToHex:
    """Tests for plotly_to_hex function."""

    def test_plotly_to_hex_hex_string(self):
        """Test conversion of plotly hex string."""
        assert plotly_to_hex("#4e79a7") == "#4e79a7"

    def test_plotly_to_hex_rgb_string(self):
        """Test conversion of plotly rgb() string."""
        assert plotly_to_hex("rgb(78, 121, 167)") == "#4e79a7"

    def test_plotly_to_hex_rgba_string(self):
        """Test conversion of plotly rgba() string."""
        assert plotly_to_hex("rgba(78, 121, 167, 1.0)") == "#4e79a7"


class TestEnsureHexPalette:
    """Tests for ensure_hex_palette function."""

    def test_ensure_hex_palette_all_hex(self):
        """Test ensuring palette with all hex colors."""
        palette = ["#4e79a7", "#f28e2b"]
        result = ensure_hex_palette(palette)
        assert result == ["#4e79a7", "#f28e2b"]

    def test_ensure_hex_palette_mixed_formats(self):
        """Test ensuring palette with mixed formats."""
        palette = ["#4e79a7", "rgb(242, 142, 43)", (0.88, 0.34, 0.35)]
        result = ensure_hex_palette(palette)
        assert len(result) == 3
        assert all(color.startswith("#") for color in result)
        assert result[0] == "#4e79a7"
        assert result[1] == "#f28e2b"


class TestRoundTripConversions:
    """Tests for round-trip color conversions."""

    def test_hex_to_rgb_to_hex_roundtrip(self):
        """Test that hex -> RGB -> hex preserves color."""
        original = "#4e79a7"
        r, g, b = hex_to_rgb(original)
        result = rgb_to_hex(r, g, b)
        assert result == original

    def test_hex_to_normalized_to_hex_roundtrip(self):
        """Test that hex -> normalized RGB -> hex preserves color (within tolerance)."""
        original = "#4e79a7"
        r, g, b = hex_to_rgb_normalized(original)
        result = rgb_normalized_to_hex(r, g, b)
        # Should match within rounding tolerance
        assert colors_match(original, result, tolerance=1)
