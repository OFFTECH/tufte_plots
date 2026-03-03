"""
Label positioning for TuftePlots.

This module contains the LabelPositioner class that handles intelligent
placement of direct labels to avoid overlaps.
"""

from typing import List, Tuple, Optional, Any


class LabelPositioner:
    """
    Handles intelligent placement of direct labels to avoid overlaps.

    This class provides methods for calculating label positions at data
    endpoints, detecting collisions between labels, and resolving overlaps
    by adjusting positions. It also ensures labels don't overlap with data
    elements like lines, bars, and points.
    """

    def __init__(self, padding: float = 0.02, data_margin: float = 0.03):
        """
        Initialize LabelPositioner with padding.

        Args:
            padding: Relative padding around labels as a fraction of the
                    plot dimensions (default: 0.02 = 2%).
            data_margin: Minimum distance between labels and data elements
                        as a fraction of plot dimensions (default: 0.03 = 3%).
        """
        self.padding = padding
        self.data_margin = data_margin

    def calculate_positions(
        self,
        endpoints: List[Tuple[float, float]],
        labels: List[str],
        bounds: Tuple[float, float, float, float],
        data_elements: Optional[List[List[Tuple[float, float]]]] = None,
    ) -> List[Tuple[float, float]]:
        """
        Calculate non-overlapping label positions.

        This method takes data endpoints and calculates positions for labels,
        attempting to place them at the endpoints while avoiding overlaps with
        both other labels and data elements.

        Args:
            endpoints: List of (x, y) coordinates representing data endpoints.
            labels: List of label strings (used for size estimation).
            bounds: Tuple of (x_min, x_max, y_min, y_max) defining plot bounds.
            data_elements: Optional list of data element paths. Each element is
                          a list of (x, y) points representing a line, bar edges,
                          or scatter points.

        Returns:
            List of (x, y) coordinates for each label.

        Raises:
            ValueError: If the number of endpoints doesn't match the number of labels.
        """
        if len(endpoints) != len(labels):
            raise ValueError(
                f"Number of endpoints ({len(endpoints)}) must match "
                f"number of labels ({len(labels)})"
            )

        # Start with positions at the endpoints
        positions = list(endpoints)

        # Estimate label sizes based on string length
        # Approximate: each character is ~0.6% of x-range, height is ~1.5% of y-range
        x_min, x_max, y_min, y_max = bounds
        x_range = x_max - x_min
        y_range = y_max - y_min

        label_sizes = [
            (len(label) * 0.006 * x_range, 0.015 * y_range) for label in labels
        ]

        # Detect and resolve collisions iteratively
        # May need multiple passes as resolving one collision can create another
        max_iterations = 100
        prev_collision_count = float("inf")
        stall_count = 0

        for iteration in range(max_iterations):
            collisions = self.detect_collisions(positions, label_sizes)

            # Also check for collisions with data elements
            if data_elements:
                data_collisions = self._detect_data_collisions(
                    positions, label_sizes, data_elements, bounds
                )
                # Add data collisions as self-collisions to trigger repositioning
                for idx in data_collisions:
                    # Mark as collision with itself to trigger adjustment
                    collisions.append((idx, idx))

            if not collisions:
                break  # No more collisions, we're done

            # Check if we're making progress
            if len(collisions) >= prev_collision_count:
                stall_count += 1
                # Allow more stalls before giving up, as sometimes progress is non-monotonic
                # especially when collision groups merge across iterations
                # Increase stall tolerance to 10 to handle complex collision scenarios
                if stall_count >= 10:
                    # Not making progress, break to avoid infinite loop
                    break
            else:
                stall_count = 0  # Reset stall counter when we make progress

            prev_collision_count = len(collisions)
            positions = self.resolve_collisions(positions, collisions, label_sizes)

        return positions

    def detect_collisions(
        self,
        positions: List[Tuple[float, float]],
        label_sizes: List[Tuple[float, float]],
    ) -> List[Tuple[int, int]]:
        """
        Return pairs of overlapping label indices.

        This method checks all pairs of labels to determine if their bounding
        boxes overlap, considering the padding around each label.

        Args:
            positions: List of (x, y) coordinates for label centers.
            label_sizes: List of (width, height) tuples for each label.

        Returns:
            List of (i, j) tuples where i and j are indices of colliding labels.

        Raises:
            ValueError: If positions and label_sizes have different lengths.
        """
        if len(positions) != len(label_sizes):
            raise ValueError(
                f"Number of positions ({len(positions)}) must match "
                f"number of label sizes ({len(label_sizes)})"
            )

        collisions = []

        # Check all pairs of labels
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if self._boxes_overlap(
                    positions[i], label_sizes[i], positions[j], label_sizes[j]
                ):
                    collisions.append((i, j))

        return collisions

    def _boxes_overlap(
        self,
        pos1: Tuple[float, float],
        size1: Tuple[float, float],
        pos2: Tuple[float, float],
        size2: Tuple[float, float],
    ) -> bool:
        """
        Check if two bounding boxes overlap.

        Args:
            pos1: (x, y) position of first label center.
            size1: (width, height) of first label.
            pos2: (x, y) position of second label center.
            size2: (width, height) of second label.

        Returns:
            True if the boxes overlap, False otherwise.
        """
        x1, y1 = pos1
        w1, h1 = size1
        x2, y2 = pos2
        w2, h2 = size2

        # Calculate bounding box edges with padding
        left1 = x1 - w1 / 2 - self.padding * w1
        right1 = x1 + w1 / 2 + self.padding * w1
        bottom1 = y1 - h1 / 2 - self.padding * h1
        top1 = y1 + h1 / 2 + self.padding * h1

        left2 = x2 - w2 / 2 - self.padding * w2
        right2 = x2 + w2 / 2 + self.padding * w2
        bottom2 = y2 - h2 / 2 - self.padding * h2
        top2 = y2 + h2 / 2 + self.padding * h2

        # Check for overlap
        # Boxes don't overlap if one is completely to the left/right/above/below the other
        # Use <= to catch touching boxes as overlapping
        if right1 <= left2 or right2 <= left1:
            return False
        if top1 <= bottom2 or top2 <= bottom1:
            return False

        return True

    def _detect_data_collisions(
        self,
        positions: List[Tuple[float, float]],
        label_sizes: List[Tuple[float, float]],
        data_elements: List[List[Tuple[float, float]]],
        bounds: Tuple[float, float, float, float],
    ) -> List[int]:
        """
        Detect which labels overlap with data elements.

        Args:
            positions: List of (x, y) coordinates for label centers.
            label_sizes: List of (width, height) tuples for each label.
            data_elements: List of data element paths (lines, bars, points).
            bounds: Tuple of (x_min, x_max, y_min, y_max) defining plot bounds.

        Returns:
            List of label indices that collide with data elements.
        """
        x_min, x_max, y_min, y_max = bounds
        x_range = x_max - x_min
        y_range = y_max - y_min

        colliding_labels = []

        for idx, (pos, size) in enumerate(zip(positions, label_sizes)):
            x, y = pos
            w, h = size

            # Calculate label bounding box with data margin
            margin_x = self.data_margin * x_range
            margin_y = self.data_margin * y_range

            left = x - w / 2 - margin_x
            right = x + w / 2 + margin_x
            bottom = y - h / 2 - margin_y
            top = y + h / 2 + margin_y

            # Check if label box intersects with any data element
            for element in data_elements:
                if self._label_intersects_element((left, right, bottom, top), element):
                    colliding_labels.append(idx)
                    break  # No need to check other elements for this label

        return colliding_labels

    def _label_intersects_element(
        self,
        label_box: Tuple[float, float, float, float],
        element: List[Tuple[float, float]],
    ) -> bool:
        """
        Check if a label bounding box intersects with a data element.

        Args:
            label_box: Tuple of (left, right, bottom, top) defining label bounds.
            element: List of (x, y) points defining the data element path.

        Returns:
            True if the label intersects the element, False otherwise.
        """
        if not element:
            return False

        left, right, bottom, top = label_box

        # Check if any point of the element is inside the label box
        for x, y in element:
            if left <= x <= right and bottom <= y <= top:
                return True

        # Check if any line segment of the element intersects the label box
        for i in range(len(element) - 1):
            x1, y1 = element[i]
            x2, y2 = element[i + 1]

            if self._line_intersects_box(x1, y1, x2, y2, label_box):
                return True

        return False

    def _line_intersects_box(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        box: Tuple[float, float, float, float],
    ) -> bool:
        """
        Check if a line segment intersects with a rectangular box.

        Uses the Liang-Barsky algorithm for line-rectangle intersection.

        Args:
            x1, y1: Start point of line segment.
            x2, y2: End point of line segment.
            box: Tuple of (left, right, bottom, top) defining box bounds.

        Returns:
            True if the line segment intersects the box, False otherwise.
        """
        left, right, bottom, top = box

        # Check if either endpoint is inside the box
        if (left <= x1 <= right and bottom <= y1 <= top) or (
            left <= x2 <= right and bottom <= y2 <= top
        ):
            return True

        # Check if line segment crosses any edge of the box
        # Use parametric line representation: P = P1 + t * (P2 - P1), where 0 <= t <= 1
        dx = x2 - x1
        dy = y2 - y1

        # Avoid division by zero
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            return False

        # Check intersection with each edge
        # Left edge (x = left)
        if abs(dx) > 1e-10:
            t = (left - x1) / dx
            if 0 <= t <= 1:
                y = y1 + t * dy
                if bottom <= y <= top:
                    return True

        # Right edge (x = right)
        if abs(dx) > 1e-10:
            t = (right - x1) / dx
            if 0 <= t <= 1:
                y = y1 + t * dy
                if bottom <= y <= top:
                    return True

        # Bottom edge (y = bottom)
        if abs(dy) > 1e-10:
            t = (bottom - y1) / dy
            if 0 <= t <= 1:
                x = x1 + t * dx
                if left <= x <= right:
                    return True

        # Top edge (y = top)
        if abs(dy) > 1e-10:
            t = (top - y1) / dy
            if 0 <= t <= 1:
                x = x1 + t * dx
                if left <= x <= right:
                    return True

        return False

    def resolve_collisions(
        self,
        positions: List[Tuple[float, float]],
        collisions: List[Tuple[int, int]],
        label_sizes: Optional[List[Tuple[float, float]]] = None,
    ) -> List[Tuple[float, float]]:
        """
        Adjust positions to eliminate overlaps.

        This method takes a list of positions and collision pairs, and adjusts
        the positions to resolve the collisions. The adjustment strategy is to
        shift colliding labels vertically with sufficient spacing.

        Args:
            positions: List of (x, y) coordinates for labels.
            collisions: List of (i, j) tuples indicating colliding label pairs.
                       Self-collisions (i, i) indicate collision with data elements.
            label_sizes: Optional list of (width, height) tuples for each label.
                        If provided, spacing will account for label dimensions.

        Returns:
            List of adjusted (x, y) coordinates.
        """
        # Create a mutable copy of positions
        adjusted = [list(pos) for pos in positions]

        # Build collision groups using union-find approach
        # Labels that collide with each other should be in the same group
        collision_set = set()
        for i, j in collisions:
            collision_set.add(i)
            if i != j:  # Don't add j for self-collisions
                collision_set.add(j)

        # Find connected components (groups of labels that collide with each other)
        groups = []
        remaining = collision_set.copy()

        while remaining:
            # Start a new group with an arbitrary label
            current_group = {remaining.pop()}
            changed = True

            # Keep adding labels that collide with any label in the current group
            while changed:
                changed = False
                for i, j in collisions:
                    # Skip self-collisions for grouping (they just mark data collisions)
                    if i == j:
                        continue
                    if i in current_group and j not in current_group:
                        current_group.add(j)
                        remaining.discard(j)
                        changed = True
                    elif j in current_group and i not in current_group:
                        current_group.add(i)
                        remaining.discard(i)
                        changed = True

            groups.append(current_group)

        # Process each group independently
        for group in groups:
            # Get indices in this group, sorted by ORIGINAL y-position (from positions parameter)
            # This ensures we maintain the original ordering intent
            group_indices = sorted(group, key=lambda idx: (positions[idx][1], idx))

            if len(group_indices) == 1:
                # Single label - check if it has a self-collision (data collision)
                idx = group_indices[0]
                # Check if this is a self-collision (data collision)
                has_self_collision = any(i == j == idx for i, j in collisions)
                if has_self_collision and label_sizes is not None:
                    # Shift label up by its height plus margin
                    h = label_sizes[idx][1]
                    adjusted[idx][1] += h * (1 + 2 * self.data_margin)
            elif len(group_indices) > 1:
                y_values = [adjusted[idx][1] for idx in group_indices]
                y_min = min(y_values)
                y_max = max(y_values)
                y_range = y_max - y_min

                # Determine minimum spacing needed between consecutive label centers
                # When labels have different sizes, we need to consider each pair
                if label_sizes is not None:
                    # Calculate spacing for each consecutive pair
                    spacings = []
                    for i in range(len(group_indices) - 1):
                        idx1 = group_indices[i]
                        idx2 = group_indices[i + 1]
                        h1 = label_sizes[idx1][1]
                        h2 = label_sizes[idx2][1]
                        # Distance between centers must be > (h1/2 + h2/2) * (1 + padding)
                        # Add extra margin (25%) to ensure strict inequality and account for rounding
                        # Increased from 15% to 25% to be more aggressive in avoiding collisions
                        pair_spacing = (h1 + h2) / 2 * (1 + 2 * self.padding) * 1.25
                        spacings.append(pair_spacing)

                    # Calculate cumulative positions
                    y_center = (y_min + y_max) / 2
                    total_span = sum(spacings)
                    y_start = y_center - total_span / 2

                    # Place first label
                    adjusted[group_indices[0]][1] = y_start

                    # Place remaining labels with appropriate spacing
                    current_y = y_start
                    for i in range(len(group_indices) - 1):
                        current_y += spacings[i]
                        adjusted[group_indices[i + 1]][1] = current_y
                else:
                    # Default minimum spacing if no label sizes provided
                    min_spacing = 1.0
                    required_range = min_spacing * (len(group_indices) - 1)
                    y_center = (y_min + y_max) / 2
                    y_start = y_center - required_range / 2

                    for i, idx in enumerate(group_indices):
                        adjusted[idx][1] = y_start + i * min_spacing

        # Convert back to tuples
        return [tuple(pos) for pos in adjusted]
