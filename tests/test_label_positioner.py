"""
Tests for LabelPositioner class.

This module tests the label positioning functionality including endpoint
placement, collision detection, and collision resolution.
"""

import pytest
from tufteplots.label_positioner import LabelPositioner


class TestLabelPositioner:
    """Test suite for LabelPositioner class."""

    def test_initialization(self):
        """Test LabelPositioner initialization with default and custom padding."""
        # Default padding
        lp = LabelPositioner()
        assert lp.padding == 0.02

        # Custom padding
        lp_custom = LabelPositioner(padding=0.05)
        assert lp_custom.padding == 0.05

    def test_calculate_positions_basic(self):
        """Test basic position calculation without collisions."""
        lp = LabelPositioner()
        endpoints = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        labels = ["A", "B", "C"]
        bounds = (0.0, 4.0, 0.0, 4.0)

        positions = lp.calculate_positions(endpoints, labels, bounds)

        assert len(positions) == 3
        assert all(isinstance(pos, tuple) for pos in positions)
        assert all(len(pos) == 2 for pos in positions)

    def test_calculate_positions_mismatch_raises_error(self):
        """Test that mismatched endpoints and labels raises ValueError."""
        lp = LabelPositioner()
        endpoints = [(1.0, 1.0), (2.0, 2.0)]
        labels = ["A", "B", "C"]  # Mismatch: 2 endpoints, 3 labels
        bounds = (0.0, 4.0, 0.0, 4.0)

        with pytest.raises(ValueError, match="must match"):
            lp.calculate_positions(endpoints, labels, bounds)

    def test_detect_collisions_no_overlap(self):
        """Test collision detection with well-separated labels."""
        lp = LabelPositioner()
        positions = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]
        label_sizes = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]

        collisions = lp.detect_collisions(positions, label_sizes)

        assert collisions == []

    def test_detect_collisions_with_overlap(self):
        """Test collision detection with overlapping labels."""
        lp = LabelPositioner()
        # These positions are very close and should collide
        positions = [(0.0, 0.0), (0.5, 0.5), (10.0, 10.0)]
        label_sizes = [(2.0, 2.0), (2.0, 2.0), (2.0, 2.0)]

        collisions = lp.detect_collisions(positions, label_sizes)

        # Should detect collision between first two labels
        assert len(collisions) > 0
        assert (0, 1) in collisions

    def test_detect_collisions_mismatch_raises_error(self):
        """Test that mismatched positions and sizes raises ValueError."""
        lp = LabelPositioner()
        positions = [(0.0, 0.0), (1.0, 1.0)]
        label_sizes = [(1.0, 1.0)]  # Mismatch

        with pytest.raises(ValueError, match="must match"):
            lp.detect_collisions(positions, label_sizes)

    def test_resolve_collisions_empty(self):
        """Test collision resolution with no collisions."""
        lp = LabelPositioner()
        positions = [(0.0, 0.0), (10.0, 10.0)]
        collisions = []

        resolved = lp.resolve_collisions(positions, collisions)

        # Positions should remain unchanged
        assert resolved == positions

    def test_resolve_collisions_adjusts_positions(self):
        """Test that collision resolution adjusts positions."""
        lp = LabelPositioner()
        positions = [(0.0, 1.0), (0.0, 1.1), (0.0, 1.2)]
        collisions = [(0, 1), (1, 2)]

        resolved = lp.resolve_collisions(positions, collisions)

        # Positions should be adjusted
        assert len(resolved) == 3
        # Y-values should be spread out evenly
        y_values = [pos[1] for pos in resolved]
        # Check that spacing is consistent (evenly distributed)
        spacing1 = y_values[1] - y_values[0]
        spacing2 = y_values[2] - y_values[1]
        # Spacings should be approximately equal (within floating point tolerance)
        assert abs(spacing1 - spacing2) < 1e-10

    def test_custom_label_positions_respected(self):
        """Test that custom positions are used as starting points."""
        lp = LabelPositioner()
        # Custom endpoint positions
        custom_endpoints = [(5.0, 5.0), (15.0, 15.0)]
        labels = ["Label1", "Label2"]
        bounds = (0.0, 20.0, 0.0, 20.0)

        positions = lp.calculate_positions(custom_endpoints, labels, bounds)

        # With no collisions, positions should match endpoints
        assert len(positions) == 2
        # X-coordinates should be preserved
        assert positions[0][0] == custom_endpoints[0][0]
        assert positions[1][0] == custom_endpoints[1][0]

    def test_endpoint_label_placement(self):
        """Test that labels are placed at data endpoints."""
        lp = LabelPositioner()
        # Simulate line plot endpoints
        endpoints = [(10.0, 5.0), (10.0, 8.0), (10.0, 3.0)]
        labels = ["Series A", "Series B", "Series C"]
        bounds = (0.0, 10.0, 0.0, 10.0)

        positions = lp.calculate_positions(endpoints, labels, bounds)

        # All labels should be at x=10 (the endpoint)
        assert all(pos[0] == 10.0 for pos in positions)
        # Y-positions should be adjusted if there were collisions
        assert len(positions) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
