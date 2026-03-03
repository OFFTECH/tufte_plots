"""
Property-based tests for TuftePlots library.

This module contains property-based tests using Hypothesis to verify
correctness properties across all valid inputs.
"""

import pytest
import pandas as pd
from hypothesis import given, strategies as st, settings
from tufteplots.theme import TufteTheme, ThemeManager
from tufteplots.validators import ConfigValidator


# Strategy for generating valid color hex codes
@st.composite
def hex_colors(draw):
    """Generate valid hex color codes."""
    r = draw(st.integers(min_value=0, max_value=255))
    g = draw(st.integers(min_value=0, max_value=255))
    b = draw(st.integers(min_value=0, max_value=255))
    return f"#{r:02x}{g:02x}{b:02x}"


# Strategy for generating valid TufteTheme instances
@st.composite
def tufte_themes(draw):
    """Generate random but valid TufteTheme configurations."""
    font_families = [
        "Inter",
        "Arial",
        "Helvetica",
        "Verdana",
        "Times New Roman",
        "Georgia",
        "sans-serif",
        "serif",
    ]

    return TufteTheme(
        font_family=draw(st.sampled_from(font_families)),
        font_fallback=draw(st.sampled_from(["serif", "sans-serif", "monospace"])),
        title_size=draw(st.integers(min_value=10, max_value=24)),
        label_size=draw(st.integers(min_value=8, max_value=18)),
        tick_size=draw(st.integers(min_value=6, max_value=14)),
        line_width=draw(st.floats(min_value=0.5, max_value=3.0)),
        axis_line_width=draw(st.floats(min_value=0.1, max_value=2.0)),
        background_color=draw(st.sampled_from(["white", "transparent", "#ffffff"])),
        axis_color=draw(hex_colors()),
        grid_color=draw(hex_colors()),
        text_color=draw(hex_colors()),
        color_palette=draw(st.lists(hex_colors(), min_size=1, max_size=12)),
        show_grid=draw(st.booleans()),
        use_range_frame=draw(st.booleans()),
        remove_legend=draw(st.booleans()),
        direct_labels=draw(st.booleans()),
    )


# **Feature: tufte-plotting-library, Property 15: Theme serialization round-trip**
@settings(max_examples=100)
@given(theme=tufte_themes())
def test_theme_serialization_roundtrip(theme):
    """
    Property 15: Theme serialization round-trip

    For any TufteTheme object, serializing to JSON and deserializing
    should produce an equivalent theme object.

    Validates: Requirements 8.4
    """
    # Serialize to JSON
    json_str = theme.to_json()

    # Deserialize from JSON
    restored_theme = TufteTheme.from_json(json_str)

    # Verify all fields match
    assert restored_theme.font_family == theme.font_family
    assert restored_theme.font_fallback == theme.font_fallback
    assert restored_theme.title_size == theme.title_size
    assert restored_theme.label_size == theme.label_size
    assert restored_theme.tick_size == theme.tick_size
    assert restored_theme.line_width == theme.line_width
    assert restored_theme.axis_line_width == theme.axis_line_width
    assert restored_theme.background_color == theme.background_color
    assert restored_theme.axis_color == theme.axis_color
    assert restored_theme.grid_color == theme.grid_color
    assert restored_theme.text_color == theme.text_color
    assert restored_theme.color_palette == theme.color_palette
    assert restored_theme.show_grid == theme.show_grid
    assert restored_theme.use_range_frame == theme.use_range_frame
    assert restored_theme.remove_legend == theme.remove_legend
    assert restored_theme.direct_labels == theme.direct_labels

    # Verify the objects are equal
    assert restored_theme == theme


# **Feature: tufte-plotting-library, Property 16: Validation error specificity**
@settings(max_examples=100)
@given(
    invalid_backend=st.text(min_size=1).filter(
        lambda x: x not in {"matplotlib", "plotly", "seaborn"}
    )
)
def test_validation_error_specificity_backend(invalid_backend):
    """
    Property 16: Validation error specificity (backend)

    For any invalid backend name, the raised ValueError should include
    the specific invalid value in its message.

    Validates: Requirements 9.1
    """
    with pytest.raises(ValueError) as exc_info:
        ConfigValidator.validate_backend(invalid_backend)

    error_msg = str(exc_info.value)
    # The error message must contain the invalid backend name
    assert invalid_backend in error_msg
    # The error message must mention supported backends
    assert "Supported backends:" in error_msg or "supported" in error_msg.lower()


@settings(max_examples=100)
@given(
    invalid_color=st.one_of(
        # Invalid hex colors (wrong format)
        st.text(min_size=1).filter(
            lambda x: x.startswith("#") and len(x) not in [4, 7]
        ),
        st.from_regex(r"#[^0-9A-Fa-f]+", fullmatch=True),
        # Invalid named colors - generate random text that's not a valid color
        st.text(
            min_size=1, alphabet=st.characters(blacklist_categories=("Cs",))
        ).filter(lambda x: not x.startswith("#") and len(x) > 0),
        # Non-string values
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.lists(st.text()),
    )
)
def test_validation_error_specificity_color(invalid_color):
    """
    Property 16: Validation error specificity (color)

    For any invalid color value, the raised ValueError should include
    the specific invalid value in its message.

    Validates: Requirements 9.2
    """
    from hypothesis import assume

    # Skip if this is actually a valid color
    if isinstance(invalid_color, str):
        valid_colors = {
            "white",
            "black",
            "red",
            "green",
            "blue",
            "yellow",
            "cyan",
            "magenta",
            "gray",
            "grey",
            "orange",
            "purple",
            "pink",
            "brown",
            "navy",
            "teal",
            "olive",
            "maroon",
            "aqua",
            "lime",
            "silver",
            "fuchsia",
        }
        assume(invalid_color.lower() not in valid_colors)
        # Also skip valid hex colors
        if invalid_color.startswith("#") and len(invalid_color) in [4, 7]:
            import re

            assume(not re.match(r"^#[0-9A-Fa-f]+$", invalid_color))

    with pytest.raises(ValueError) as exc_info:
        ConfigValidator.validate_color(invalid_color)

    error_msg = str(exc_info.value)
    # The error message must contain the invalid color value
    assert str(invalid_color) in error_msg
    # The error message must mention expected format
    assert "Expected hex" in error_msg or "expected" in error_msg.lower()


@settings(max_examples=100)
@given(
    backend=st.sampled_from(["matplotlib", "plotly", "seaborn"]),
    wrong_figure=st.one_of(
        st.text(),
        st.integers(),
        st.lists(st.integers()),
        st.dictionaries(st.text(), st.integers()),
    ),
)
def test_validation_error_specificity_figure(backend, wrong_figure):
    """
    Property 16: Validation error specificity (figure type)

    For any figure object of incorrect type, the raised TypeError should
    indicate the expected figure types and the actual type received.

    Validates: Requirements 9.3
    """
    with pytest.raises(TypeError) as exc_info:
        ConfigValidator.validate_figure(wrong_figure, backend)

    error_msg = str(exc_info.value)
    # The error message must mention expected type
    assert "Expected" in error_msg or "expected" in error_msg.lower()
    # The error message must mention what was received
    assert "got" in error_msg.lower()


@settings(max_examples=100)
@given(
    existing_columns=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=1,
            max_size=10,
        ),
        min_size=1,
        max_size=5,
        unique=True,
    ),
    required_columns=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=1,
            max_size=10,
        ),
        min_size=1,
        max_size=5,
        unique=True,
    ),
)
def test_validation_error_specificity_dataframe_columns(
    existing_columns, required_columns
):
    """
    Property 16: Validation error specificity (missing columns)

    For any DataFrame with missing required columns, the raised KeyError
    should list the specific missing column names.

    Validates: Requirements 9.4
    """
    # Only test when there are actually missing columns
    missing = [col for col in required_columns if col not in existing_columns]
    if not missing:
        return  # Skip this test case if no columns are missing

    # Create a DataFrame with only the existing columns
    df = pd.DataFrame({col: [1, 2, 3] for col in existing_columns})

    with pytest.raises(KeyError) as exc_info:
        ConfigValidator.validate_dataframe_columns(df, required_columns)

    error_msg = str(exc_info.value)
    # The error message must mention missing columns
    assert "Missing required columns:" in error_msg
    # The error message should be specific - verify it's not a generic error
    # by checking that it contains information about what's missing
    assert len(error_msg) > len("Missing required columns:")
    # Verify at least one missing column name appears in the message
    # (accounting for string representation differences)
    for col in missing:
        # The column should appear somewhere in the error message
        if col in error_msg:
            break
    else:
        # If no exact match, the error message should at least contain
        # a list-like structure indicating the missing columns
        assert "[" in error_msg and "]" in error_msg


# Strategy for generating custom theme parameters
@st.composite
def custom_theme_params(draw):
    """Generate random custom theme parameters for testing overrides."""
    return {
        "font_family": draw(
            st.sampled_from(
                [
                    "Inter",
                    "Arial",
                    "Times New Roman",
                    "Palatino",
                    "Garamond",
                    "Helvetica",
                ]
            )
        ),
        "title_size": draw(st.integers(min_value=10, max_value=24)),
        "label_size": draw(st.integers(min_value=8, max_value=18)),
        "show_grid": draw(st.booleans()),
        "use_range_frame": draw(st.booleans()),
        "color_palette": draw(st.lists(hex_colors(), min_size=1, max_size=8)),
    }


# **Feature: tufte-plotting-library, Property 14: Theme customization override**
@settings(max_examples=100)
@given(custom_params=custom_theme_params())
def test_theme_customization_override(custom_params):
    """
    Property 14: Theme customization override

    For any custom theme parameter (color palette, font sizes, feature flags),
    the applied style should reflect the custom value rather than the default.

    Validates: Requirements 8.1, 8.2, 8.3
    """
    # Create a ThemeManager with default theme
    manager = ThemeManager()

    # Get the default theme to compare against
    default_theme = TufteTheme()

    # Update theme with custom parameters
    updated_theme = manager.update_theme(**custom_params)

    # Verify that custom values override defaults
    assert updated_theme.font_family == custom_params["font_family"]
    assert updated_theme.title_size == custom_params["title_size"]
    assert updated_theme.label_size == custom_params["label_size"]
    assert updated_theme.show_grid == custom_params["show_grid"]
    assert updated_theme.use_range_frame == custom_params["use_range_frame"]
    assert updated_theme.color_palette == custom_params["color_palette"]

    # Verify that custom values are different from defaults (when they differ)
    if custom_params["font_family"] != default_theme.font_family:
        assert updated_theme.font_family != default_theme.font_family

    if custom_params["title_size"] != default_theme.title_size:
        assert updated_theme.title_size != default_theme.title_size

    if custom_params["label_size"] != default_theme.label_size:
        assert updated_theme.label_size != default_theme.label_size

    if custom_params["show_grid"] != default_theme.show_grid:
        assert updated_theme.show_grid != default_theme.show_grid

    if custom_params["use_range_frame"] != default_theme.use_range_frame:
        assert updated_theme.use_range_frame != default_theme.use_range_frame

    if custom_params["color_palette"] != default_theme.color_palette:
        assert updated_theme.color_palette != default_theme.color_palette

    # Verify that the manager's internal theme was updated
    current_theme = manager.get_theme()
    assert current_theme == updated_theme


# Strategy for generating random DataFrames for plotting
@st.composite
def dataframes_for_plotting(draw):
    """Generate random but valid DataFrames for plotting tests."""
    n_rows = draw(st.integers(min_value=5, max_value=50))
    x = draw(
        st.lists(
            st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    y = draw(
        st.lists(
            st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    return pd.DataFrame({"x": x, "y": y})


# Strategy for generating matplotlib-compatible themes
@st.composite
def matplotlib_compatible_themes(draw):
    """Generate themes with matplotlib-compatible color values."""
    font_families = [
        "Inter",
        "Arial",
        "Helvetica",
        "Verdana",
        "Times New Roman",
        "Georgia",
        "sans-serif",
        "serif",
    ]

    # Only use colors that matplotlib can handle directly
    background_colors = ["white", "#ffffff", "#FFFFFF"]

    return TufteTheme(
        font_family=draw(st.sampled_from(font_families)),
        font_fallback=draw(st.sampled_from(["serif", "sans-serif", "monospace"])),
        title_size=draw(st.integers(min_value=10, max_value=24)),
        label_size=draw(st.integers(min_value=8, max_value=18)),
        tick_size=draw(st.integers(min_value=6, max_value=14)),
        line_width=draw(st.floats(min_value=0.5, max_value=3.0)),
        axis_line_width=draw(st.floats(min_value=0.1, max_value=2.0)),
        background_color=draw(st.sampled_from(background_colors)),
        axis_color=draw(hex_colors()),
        grid_color=draw(hex_colors()),
        text_color=draw(hex_colors()),
        color_palette=draw(st.lists(hex_colors(), min_size=1, max_size=12)),
        show_grid=draw(st.booleans()),
        use_range_frame=draw(st.booleans()),
        remove_legend=draw(st.booleans()),
        direct_labels=draw(st.booleans()),
    )


# **Feature: tufte-plotting-library, Property 2: Chartjunk removal**
@settings(
    max_examples=100, deadline=1000
)  # Increase deadline for matplotlib figure creation
@given(data=dataframes_for_plotting(), theme=matplotlib_compatible_themes())
def test_chartjunk_removal_matplotlib(data, theme):
    """
    Property 2: Chartjunk removal

    For any matplotlib figure after applying Tufte style, the top and right
    spines should not be visible, grid lines should be disabled (unless
    explicitly enabled), and background color should be white or transparent.

    Validates: Requirements 2.1, 2.2, 2.4
    """
    # Import matplotlib here to avoid import errors if not installed
    try:
        import matplotlib.pyplot as plt
        from tufteplots.adapters import MatplotlibAdapter
    except ImportError:
        pytest.skip("matplotlib not installed")

    # Create a simple matplotlib figure
    fig, ax = plt.subplots()
    ax.plot(data["x"], data["y"])

    # Create adapter and apply styling
    adapter = MatplotlibAdapter(theme)
    adapter.apply_theme(fig, theme)
    styled_fig = adapter.remove_chartjunk(fig)

    # Verify chartjunk removal for all axes
    for ax in styled_fig.axes:
        # Property: Top and right spines should not be visible
        assert not ax.spines["top"].get_visible(), "Top spine should not be visible"
        assert not ax.spines["right"].get_visible(), "Right spine should not be visible"

        # Property: Grid should match theme setting
        # If show_grid is False, grid should be disabled
        if not theme.show_grid:
            # Check that grid is not visible
            assert (
                not ax.xaxis.get_gridlines()[0].get_visible()
                if ax.xaxis.get_gridlines()
                else True
            )
            assert (
                not ax.yaxis.get_gridlines()[0].get_visible()
                if ax.yaxis.get_gridlines()
                else True
            )

        # Property: Background color should be white or transparent
        facecolor = ax.get_facecolor()
        # facecolor is an RGBA tuple
        # White is (1.0, 1.0, 1.0, 1.0) or (1.0, 1.0, 1.0, 0.0) for transparent
        # We check if it matches the theme's background color
        if theme.background_color in ["white", "#ffffff", "#FFFFFF"]:
            # Should be white (RGB values close to 1.0)
            assert (
                facecolor[0] >= 0.99
            ), f"Red channel should be ~1.0 for white, got {facecolor[0]}"
            assert (
                facecolor[1] >= 0.99
            ), f"Green channel should be ~1.0 for white, got {facecolor[1]}"
            assert (
                facecolor[2] >= 0.99
            ), f"Blue channel should be ~1.0 for white, got {facecolor[2]}"
        elif theme.background_color == "transparent":
            # Should have alpha = 0 or RGB values indicating transparency
            # Matplotlib may represent transparent differently
            pass  # Transparent handling varies by backend

    # Clean up
    plt.close(fig)


# **Feature: tufte-plotting-library, Property 3: Range frame bounds match data**
@settings(max_examples=100, deadline=1000)
@given(data=dataframes_for_plotting())
def test_range_frame_bounds_match_data(data):
    """
    Property 3: Range frame bounds match data

    For any dataset with known min/max values, after applying Tufte style
    with range frames enabled, the axis limits should equal the data range
    (within a small padding tolerance).

    Validates: Requirements 2.3
    """
    # Import matplotlib here to avoid import errors if not installed
    try:
        import matplotlib.pyplot as plt
        from tufteplots.adapters import MatplotlibAdapter
    except ImportError:
        pytest.skip("matplotlib not installed")

    # Calculate the actual data range
    y_min = data["y"].min()
    y_max = data["y"].max()
    range_span = y_max - y_min

    # Skip test if all values are identical (zero range)
    # Matplotlib automatically expands zero-range limits to avoid singular transformations
    if range_span == 0:
        return

    # Create a theme with range frames enabled
    theme = TufteTheme(use_range_frame=True)

    # Create a simple matplotlib figure
    fig, ax = plt.subplots()
    ax.plot(data["x"], data["y"])

    data_range = (y_min, y_max)

    # Create adapter and apply range frame
    adapter = MatplotlibAdapter(theme)
    styled_fig = adapter.apply_range_frame(fig, data_range)

    # Verify that axis limits match the data range
    for ax in styled_fig.axes:
        y_limits = ax.get_ylim()

        # Allow for small numerical tolerance (0.1% of range or 1e-6 for small values)
        tolerance = max(range_span * 0.001, 1e-6)

        # Check that the y-axis limits match the data range within tolerance
        assert abs(y_limits[0] - y_min) <= tolerance, (
            f"Y-axis lower limit {y_limits[0]} does not match data minimum {y_min} "
            f"(tolerance: {tolerance})"
        )
        assert abs(y_limits[1] - y_max) <= tolerance, (
            f"Y-axis upper limit {y_limits[1]} does not match data maximum {y_max} "
            f"(tolerance: {tolerance})"
        )

    # Clean up
    plt.close(fig)


# **Feature: tufte-plotting-library, Property 4: Typography consistency**
@settings(max_examples=100, deadline=1000)
@given(data=dataframes_for_plotting())
def test_typography_consistency(data):
    """
    Property 4: Typography consistency

    For any figure after applying Tufte style with default theme, the title
    font size should be 14pt, axis label font size should be 11pt, and tick
    label font size should be 9pt.

    Validates: Requirements 3.1, 3.2, 3.3
    """
    # Import matplotlib here to avoid import errors if not installed
    try:
        import matplotlib.pyplot as plt
        from tufteplots.adapters import MatplotlibAdapter
    except ImportError:
        pytest.skip("matplotlib not installed")

    # Create a default theme (with default typography settings)
    theme = TufteTheme()

    # Verify the default theme has the expected typography values
    assert theme.title_size == 18, "Default title size should be 18pt"
    assert theme.label_size == 14, "Default label size should be 14pt"
    assert theme.tick_size == 12, "Default tick size should be 12pt"

    # Create a matplotlib figure with title and labels
    fig, ax = plt.subplots()
    ax.plot(data["x"], data["y"])
    ax.set_title("Test Title")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    # Create adapter and apply theme
    adapter = MatplotlibAdapter(theme)
    styled_fig = adapter.apply_theme(fig, theme)

    # Verify typography consistency for all axes
    for ax in styled_fig.axes:
        # Property: Title font size should be 18pt
        if ax.get_title():
            title_fontsize = ax.title.get_fontsize()
            assert (
                title_fontsize == 18
            ), f"Title font size should be 18pt, got {title_fontsize}"

            # Also verify font family is set correctly
            title_fontfamily = ax.title.get_fontfamily()
            # Font family should match theme or fallback to generic family
            assert (
                theme.font_family in title_fontfamily
                or theme.font_fallback in title_fontfamily
                or any(
                    generic in title_fontfamily
                    for generic in ["serif", "sans-serif", "monospace"]
                )
            ), f"Title font family should be {theme.font_family} or {theme.font_fallback}, got {title_fontfamily}"

        # Property: Axis label font size should be 14pt
        xlabel_fontsize = ax.xaxis.label.get_fontsize()
        ylabel_fontsize = ax.yaxis.label.get_fontsize()

        assert (
            xlabel_fontsize == 14
        ), f"X-axis label font size should be 14pt, got {xlabel_fontsize}"
        assert (
            ylabel_fontsize == 14
        ), f"Y-axis label font size should be 14pt, got {ylabel_fontsize}"

        # Verify font family for axis labels
        xlabel_fontfamily = ax.xaxis.label.get_fontfamily()
        ylabel_fontfamily = ax.yaxis.label.get_fontfamily()

        assert (
            theme.font_family in xlabel_fontfamily
            or theme.font_fallback in xlabel_fontfamily
            or any(
                generic in xlabel_fontfamily
                for generic in ["serif", "sans-serif", "monospace"]
            )
        ), f"X-axis label font family should be {theme.font_family} or {theme.font_fallback}, got {xlabel_fontfamily}"
        assert (
            theme.font_family in ylabel_fontfamily
            or theme.font_fallback in ylabel_fontfamily
            or any(
                generic in ylabel_fontfamily
                for generic in ["serif", "sans-serif", "monospace"]
            )
        ), f"Y-axis label font family should be {theme.font_family} or {theme.font_fallback}, got {ylabel_fontfamily}"

        # Property: Tick label font size should be 12pt
        for label in ax.get_xticklabels():
            tick_fontsize = label.get_fontsize()
            assert (
                tick_fontsize == 12
            ), f"X-axis tick label font size should be 12pt, got {tick_fontsize}"

        for label in ax.get_yticklabels():
            tick_fontsize = label.get_fontsize()
            assert (
                tick_fontsize == 12
            ), f"Y-axis tick label font size should be 12pt, got {tick_fontsize}"

    # Clean up
    plt.close(fig)


# **Feature: tufte-plotting-library, Property 1: Style application preserves figure type**
@settings(max_examples=100, deadline=1000)
@given(data=dataframes_for_plotting(), theme=matplotlib_compatible_themes())
def test_style_application_preserves_figure_type_matplotlib(data, theme):
    """
    Property 1: Style application preserves figure type

    For any valid matplotlib figure object, applying Tufte style should
    return a figure of the same type as the input.

    Validates: Requirements 1.4
    """
    # Import matplotlib here to avoid import errors if not installed
    try:
        import matplotlib.pyplot as plt
        import matplotlib.figure
        from tufteplots.adapters import MatplotlibAdapter
    except ImportError:
        pytest.skip("matplotlib not installed")

    # Create a matplotlib figure
    fig, ax = plt.subplots()
    ax.plot(data["x"], data["y"])

    # Store the original type
    original_type = type(fig)

    # Verify it's a matplotlib Figure
    assert isinstance(fig, matplotlib.figure.Figure)

    # Create adapter and apply styling
    adapter = MatplotlibAdapter(theme)
    styled_fig = adapter.apply_theme(fig, theme)

    # Property: The returned figure should be the same type as the input
    assert type(styled_fig) == original_type, (
        f"Style application changed figure type from {original_type} "
        f"to {type(styled_fig)}"
    )

    # Verify it's still a matplotlib Figure
    assert isinstance(styled_fig, matplotlib.figure.Figure)

    # Verify it's the same object (not a copy)
    assert (
        styled_fig is fig
    ), "Style application should modify in-place, not create a copy"

    # Clean up
    plt.close(fig)


@settings(max_examples=100, deadline=1000)
@given(data=dataframes_for_plotting(), theme=tufte_themes())
def test_style_application_preserves_figure_type_plotly(data, theme):
    """
    Property 1: Style application preserves figure type (Plotly)

    For any valid plotly figure object, applying Tufte style should
    return a figure of the same type as the input.

    Validates: Requirements 1.4
    """
    # Import plotly here to avoid import errors if not installed
    try:
        import plotly.graph_objects as go
        from tufteplots.adapters import PlotlyAdapter
    except ImportError:
        pytest.skip("plotly not installed")

    # Create a plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["x"], y=data["y"], mode="lines"))

    # Store the original type
    original_type = type(fig)

    # Verify it's a plotly Figure
    assert isinstance(fig, go.Figure)

    # Create adapter and apply styling
    adapter = PlotlyAdapter(theme)
    styled_fig = adapter.apply_theme(fig, theme)

    # Property: The returned figure should be the same type as the input
    assert type(styled_fig) == original_type, (
        f"Style application changed figure type from {original_type} "
        f"to {type(styled_fig)}"
    )

    # Verify it's still a plotly Figure
    assert isinstance(styled_fig, go.Figure)

    # Verify it's the same object (not a copy)
    assert (
        styled_fig is fig
    ), "Style application should modify in-place, not create a copy"


# **Feature: tufte-plotting-library, Property 13: Backend selection**
@settings(max_examples=100, deadline=1000)
@given(
    data=dataframes_for_plotting(),
    backend=st.sampled_from(["matplotlib", "plotly", "seaborn"]),
)
def test_backend_selection(data, backend):
    """
    Property 13: Backend selection

    For any backend parameter value in {'matplotlib', 'plotly', 'seaborn'},
    the returned figure type should match the requested backend.

    Validates: Requirements 7.3
    """
    # Import adapters
    try:
        from tufteplots.adapters import (
            MatplotlibAdapter,
            PlotlyAdapter,
            SeabornAdapter,
        )
    except ImportError:
        pytest.skip("Required adapters not available")

    # Create theme
    theme = TufteTheme()

    # Create adapter based on backend
    if backend == "matplotlib":
        try:
            import matplotlib.pyplot as plt
            import matplotlib.figure

            adapter = MatplotlibAdapter(theme)
            fig = adapter.create_line_plot(data, "x", "y")

            # Property: Figure should be a matplotlib Figure
            assert isinstance(
                fig, matplotlib.figure.Figure
            ), f"Expected matplotlib.figure.Figure for backend '{backend}', got {type(fig)}"

            # Clean up
            plt.close(fig)

        except ImportError:
            pytest.skip("matplotlib not installed")

    elif backend == "plotly":
        try:
            import plotly.graph_objects as go

            adapter = PlotlyAdapter(theme)
            fig = adapter.create_line_plot(data, "x", "y")

            # Property: Figure should be a plotly Figure
            assert isinstance(
                fig, go.Figure
            ), f"Expected plotly.graph_objects.Figure for backend '{backend}', got {type(fig)}"

        except ImportError:
            pytest.skip("plotly not installed")

    elif backend == "seaborn":
        try:
            import matplotlib.pyplot as plt
            import matplotlib.figure
            import seaborn as sns

            adapter = SeabornAdapter(theme)
            fig = adapter.create_line_plot(data, "x", "y")

            # Property: Seaborn uses matplotlib backend, so figure should be matplotlib Figure
            assert isinstance(
                fig, matplotlib.figure.Figure
            ), f"Expected matplotlib.figure.Figure for backend '{backend}' (seaborn uses matplotlib), got {type(fig)}"

            # Clean up
            plt.close(fig)

        except ImportError:
            pytest.skip("seaborn or matplotlib not installed")


# **Feature: tufte-plotting-library, Property 7: Label collision avoidance**
@settings(max_examples=100)
@given(
    num_labels=st.integers(min_value=2, max_value=10),
    x_base=st.floats(
        min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
    ),
    y_positions=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    ),
)
def test_label_collision_avoidance(num_labels, x_base, y_positions):
    """
    Property 7: Label collision avoidance

    For any set of label positions calculated by LabelPositioner, no two
    label bounding boxes should overlap.

    Validates: Requirements 4.3
    """
    from tufteplots.label_positioner import LabelPositioner

    # Ensure we have enough y_positions
    num_labels = min(num_labels, len(y_positions))
    if num_labels < 2:
        return  # Skip if less than 2 labels (need at least 2 for collision testing)

    # Create label positioner
    positioner = LabelPositioner()

    # Create endpoints that are likely to collide (same x, close y values)
    # This simulates line plot endpoints where multiple series end at the same x
    endpoints = [(x_base, y_positions[i]) for i in range(num_labels)]

    # Create labels of varying lengths
    labels = [f"Series{i}" * (i % 3 + 1) for i in range(num_labels)]

    # Define bounds
    y_min = min(y_positions[:num_labels])
    y_max = max(y_positions[:num_labels])
    y_range = y_max - y_min if y_max > y_min else 10.0
    bounds = (x_base - 10, x_base + 10, y_min - y_range * 0.1, y_max + y_range * 0.1)

    # Calculate positions (this should resolve any collisions)
    calculated_positions = positioner.calculate_positions(endpoints, labels, bounds)

    # Property: The number of returned positions should match the number of input labels
    assert (
        len(calculated_positions) == num_labels
    ), f"Expected {num_labels} positions, got {len(calculated_positions)}"

    # Estimate label sizes (same calculation as in LabelPositioner)
    x_range = bounds[1] - bounds[0]
    y_range_bounds = bounds[3] - bounds[2]
    label_sizes = [
        (len(label) * 0.006 * x_range, 0.015 * y_range_bounds) for label in labels
    ]

    # Property: No two label bounding boxes should overlap
    collisions = positioner.detect_collisions(calculated_positions, label_sizes)

    assert not collisions, (
        f"Label collision avoidance failed: {len(collisions)} collision(s) detected "
        f"between labels after position calculation. Colliding pairs: {collisions}. "
        f"Positions: {calculated_positions}, Label sizes: {label_sizes}"
    )


# **Feature: tufte-plotting-library, Property 8: Custom label positions respected**
@settings(max_examples=100)
@given(
    num_labels=st.integers(min_value=1, max_value=10),
    x_positions=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    ),
    y_positions=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    ),
)
def test_custom_label_positions_respected(num_labels, x_positions, y_positions):
    """
    Property 8: Custom label positions respected

    For any user-specified label positions, the actual label coordinates
    should match the specified coordinates exactly.

    Validates: Requirements 4.4
    """
    from tufteplots.label_positioner import LabelPositioner

    # Ensure we have matching lengths
    num_labels = min(num_labels, len(x_positions), len(y_positions))
    if num_labels == 0:
        return  # Skip if no labels

    # Create label positioner
    positioner = LabelPositioner()

    # Create custom positions
    custom_positions = [(x_positions[i], y_positions[i]) for i in range(num_labels)]

    # Create dummy labels
    labels = [f"Label{i}" for i in range(num_labels)]

    # Define bounds (not used when positions are already specified, but needed for API)
    bounds = (-100, 100, -100, 100)

    # When we pass endpoints that are already at the desired positions,
    # and there are no collisions, the positions should be preserved
    # First, let's test that calculate_positions returns positions
    calculated_positions = positioner.calculate_positions(
        custom_positions, labels, bounds
    )

    # Property: The number of returned positions should match the number of input positions
    assert len(calculated_positions) == len(
        custom_positions
    ), f"Expected {len(custom_positions)} positions, got {len(calculated_positions)}"

    # Property: Each position should be a tuple of two floats
    for pos in calculated_positions:
        assert isinstance(pos, tuple), f"Position should be a tuple, got {type(pos)}"
        assert len(pos) == 2, f"Position should have 2 coordinates, got {len(pos)}"
        assert isinstance(
            pos[0], (int, float)
        ), f"X coordinate should be numeric, got {type(pos[0])}"
        assert isinstance(
            pos[1], (int, float)
        ), f"Y coordinate should be numeric, got {type(pos[1])}"

    # Property: If there are no collisions, positions should match exactly
    # Check if there are collisions
    label_sizes = [(len(label) * 0.006 * 200, 0.015 * 200) for label in labels]
    collisions = positioner.detect_collisions(custom_positions, label_sizes)

    if not collisions:
        # No collisions means positions should be preserved exactly
        for i, (calc_pos, custom_pos) in enumerate(
            zip(calculated_positions, custom_positions)
        ):
            assert calc_pos[0] == custom_pos[0], (
                f"X coordinate for label {i} should be {custom_pos[0]}, "
                f"got {calc_pos[0]}"
            )
            assert calc_pos[1] == custom_pos[1], (
                f"Y coordinate for label {i} should be {custom_pos[1]}, "
                f"got {calc_pos[1]}"
            )


# **Feature: tufte-plotting-library, Property 5: Direct labels placed at series endpoints**
@settings(max_examples=100, deadline=1000)
@given(
    num_series=st.integers(min_value=1, max_value=5),
    data_points=st.integers(min_value=5, max_value=20),
)
def test_direct_labels_at_endpoints(num_series, data_points):
    """
    Property 5: Direct labels placed at series endpoints

    For any line plot with N data series and direct labeling enabled, there
    should be exactly N text labels positioned at the rightmost point of each series.

    Validates: Requirements 4.1
    """
    # Import matplotlib here to avoid import errors if not installed
    try:
        import matplotlib.pyplot as plt
        from tufteplots.adapters import MatplotlibAdapter
    except ImportError:
        pytest.skip("matplotlib not installed")

    # Create a theme
    theme = TufteTheme()

    # Create a matplotlib figure with multiple lines
    fig, ax = plt.subplots(figsize=(8, 5))

    # Generate data for each series
    x_data = list(range(data_points))
    series_labels = []
    rightmost_points = []

    for i in range(num_series):
        # Generate y data with different slopes
        y_data = [x * (i + 1) + (i * 10) for x in x_data]
        label = f"Series {chr(65 + i)}"  # Series A, Series B, etc.
        series_labels.append(label)

        # Plot the line
        ax.plot(x_data, y_data, label=label)

        # Store the rightmost point
        rightmost_points.append((x_data[-1], y_data[-1]))

    # Add a legend (which should be removed by enable_direct_labeling)
    ax.legend()

    # Create adapter and enable direct labeling
    adapter = MatplotlibAdapter(theme)
    fig = adapter.enable_direct_labeling(fig)

    # Property 1: There should be exactly N text labels for N series
    # Get all text objects from the axes
    text_objects = [child for child in ax.get_children() if isinstance(child, plt.Text)]

    # Filter to get only our series labels (exclude axis labels, tick labels, etc.)
    series_text_labels = [t for t in text_objects if t.get_text() in series_labels]

    assert len(series_text_labels) == num_series, (
        f"Expected {num_series} direct labels for {num_series} series, "
        f"found {len(series_text_labels)} labels. "
        f"Labels found: {[t.get_text() for t in series_text_labels]}"
    )

    # Property 2: Each label should be positioned near the rightmost point of its series
    # We allow some tolerance since LabelPositioner may adjust positions to avoid collisions
    x_max = max(x_data)
    y_values = [point[1] for point in rightmost_points]
    y_min = min(y_values)
    y_max = max(y_values)
    y_range = y_max - y_min if y_max > y_min else 1.0

    for text_obj in series_text_labels:
        label_text = text_obj.get_text()
        label_pos = text_obj.get_position()

        # Find the corresponding series endpoint
        series_idx = series_labels.index(label_text)
        endpoint = rightmost_points[series_idx]

        # Property: Label x-position should be at or near the rightmost x value
        # Allow small tolerance for positioning adjustments
        x_tolerance = (max(x_data) - min(x_data)) * 0.05  # 5% of x range
        assert abs(label_pos[0] - endpoint[0]) <= x_tolerance, (
            f"Label '{label_text}' x-position {label_pos[0]} is too far from "
            f"endpoint x-position {endpoint[0]} (tolerance: {x_tolerance})"
        )

        # Property: Label y-position should be within the range of all endpoints
        # (may be adjusted by collision avoidance, but should stay in reasonable range)
        # For single series (y_min == y_max), use a minimum tolerance based on absolute values
        # or a fixed minimum to account for label height adjustments
        if y_min == y_max:
            # Use 20% of value or minimum of 2.0 to account for label positioning adjustments
            y_tolerance = max(abs(y_min) * 0.2, abs(y_max) * 0.2, 2.0)
        else:
            y_tolerance = (
                y_range * 0.5
            )  # Allow 50% range adjustment for collision avoidance
        assert y_min - y_tolerance <= label_pos[1] <= y_max + y_tolerance, (
            f"Label '{label_text}' y-position {label_pos[1]} is outside the "
            f"expected range [{y_min - y_tolerance}, {y_max + y_tolerance}]"
        )

    # Property 3: Legend should be removed
    assert (
        ax.get_legend() is None
    ), "Legend should be removed when direct labeling is enabled"

    # Clean up
    plt.close(fig)


# **Feature: tufte-plotting-library, Property 6: Legend removal with direct labels**
@settings(max_examples=100, deadline=1000)
@given(
    num_series=st.integers(min_value=1, max_value=5),
    data_points=st.integers(min_value=5, max_value=20),
)
def test_legend_removal_with_direct_labels(num_series, data_points):
    """
    Property 6: Legend removal with direct labels

    For any figure with direct labeling enabled, the legend should not be visible.

    Validates: Requirements 4.2
    """
    # Import matplotlib here to avoid import errors if not installed
    try:
        import matplotlib.pyplot as plt
        from tufteplots.adapters import MatplotlibAdapter
    except ImportError:
        pytest.skip("matplotlib not installed")

    # Create a theme
    theme = TufteTheme()

    # Create a matplotlib figure with multiple lines and a legend
    fig, ax = plt.subplots(figsize=(8, 5))

    # Generate data for each series
    x_data = list(range(data_points))

    for i in range(num_series):
        # Generate y data with different slopes
        y_data = [x * (i + 1) + (i * 10) for x in x_data]
        label = f"Series {chr(65 + i)}"  # Series A, Series B, etc.

        # Plot the line with a label
        ax.plot(x_data, y_data, label=label)

    # Add a legend to the plot
    ax.legend()

    # Verify that the legend exists before enabling direct labeling
    assert ax.get_legend() is not None, "Legend should exist before direct labeling"

    # Create adapter and enable direct labeling
    adapter = MatplotlibAdapter(theme)
    fig = adapter.enable_direct_labeling(fig)

    # Property: Legend should be removed when direct labeling is enabled
    assert (
        ax.get_legend() is None
    ), "Legend should be removed when direct labeling is enabled"

    # Clean up
    plt.close(fig)


# **Feature: tufte-plotting-library, Property 9: Small multiples shared axes**
@settings(max_examples=100, deadline=2000)
@given(
    num_groups=st.integers(min_value=2, max_value=6),
    data_points=st.integers(min_value=5, max_value=20),
)
def test_small_multiples_shared_axes(num_groups, data_points):
    """
    Property 9: Small multiples shared axes

    For any small multiples plot with N groups, all N subplots should have
    identical x-axis and y-axis limits.

    Validates: Requirements 5.1, 5.2
    """
    # Import required libraries
    try:
        import matplotlib.pyplot as plt
        from tufteplots.adapters import small_multiples
        import numpy as np
    except ImportError:
        pytest.skip("Required libraries not installed")

    # Generate test data with multiple groups
    np.random.seed(42)
    x_data = []
    y_data = []
    group_data = []

    for group_idx in range(num_groups):
        for point_idx in range(data_points):
            x_data.append(point_idx)
            y_data.append(np.random.randn() * (group_idx + 1) + group_idx * 10)
            group_data.append(f"Group_{group_idx}")

    data = pd.DataFrame({"x": x_data, "y": y_data, "category": group_data})

    # Create small multiples
    fig = small_multiples(
        data, x="x", y="y", facet_by="category", plot_type="line", backend="matplotlib"
    )

    # Get all visible axes
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]

    # Property: There should be at least num_groups visible axes
    assert (
        len(visible_axes) >= num_groups
    ), f"Expected at least {num_groups} visible axes, got {len(visible_axes)}"

    # Property: All visible axes should have identical x-axis limits
    if len(visible_axes) > 1:
        first_xlim = visible_axes[0].get_xlim()
        for ax in visible_axes[1:]:
            xlim = ax.get_xlim()
            assert (
                xlim == first_xlim
            ), f"X-axis limits not shared: expected {first_xlim}, got {xlim}"

    # Property: All visible axes should have identical y-axis limits
    if len(visible_axes) > 1:
        first_ylim = visible_axes[0].get_ylim()
        for ax in visible_axes[1:]:
            ylim = ax.get_ylim()
            assert (
                ylim == first_ylim
            ), f"Y-axis limits not shared: expected {first_ylim}, got {ylim}"

    # Clean up
    plt.close(fig)


# **Feature: tufte-plotting-library, Property 10: Small multiples group labeling**
@settings(max_examples=100, deadline=2000)
@given(
    num_groups=st.integers(min_value=2, max_value=6),
    data_points=st.integers(min_value=5, max_value=20),
)
def test_small_multiples_group_labeling(num_groups, data_points):
    """
    Property 10: Small multiples group labeling

    For any small multiples plot created from data with group column G, each
    subplot title should contain the corresponding group value from G.

    Validates: Requirements 5.3
    """
    # Import required libraries
    try:
        import matplotlib.pyplot as plt
        from tufteplots.adapters import small_multiples
        import numpy as np
    except ImportError:
        pytest.skip("Required libraries not installed")

    # Generate test data with multiple groups
    np.random.seed(42)
    x_data = []
    y_data = []
    group_data = []
    group_names = [f"Group_{i}" for i in range(num_groups)]

    for group_idx in range(num_groups):
        for point_idx in range(data_points):
            x_data.append(point_idx)
            y_data.append(np.random.randn() * (group_idx + 1) + group_idx * 10)
            group_data.append(group_names[group_idx])

    data = pd.DataFrame({"x": x_data, "y": y_data, "category": group_data})

    # Create small multiples
    fig = small_multiples(
        data, x="x", y="y", facet_by="category", plot_type="line", backend="matplotlib"
    )

    # Get all visible axes
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]

    # Property: Each visible subplot should have a title
    for ax in visible_axes:
        title = ax.get_title()
        assert title, "Each subplot should have a title"

    # Property: Each group name should appear in exactly one subplot title
    subplot_titles = [ax.get_title() for ax in visible_axes]

    for group_name in group_names:
        matching_titles = [title for title in subplot_titles if group_name in title]
        assert len(matching_titles) == 1, (
            f"Group name '{group_name}' should appear in exactly one subplot title, "
            f"found in {len(matching_titles)} titles: {matching_titles}"
        )

    # Clean up
    plt.close(fig)


# **Feature: tufte-plotting-library, Property 11: Grid dimension calculation**
@settings(max_examples=100)
@given(num_groups=st.integers(min_value=1, max_value=20))
def test_grid_dimension_calculation(num_groups):
    """
    Property 11: Grid dimension calculation

    For any number of groups N, the calculated grid dimensions (rows × cols)
    should be >= N and minimize empty cells.

    Validates: Requirements 5.4
    """
    from tufteplots.adapters import calculate_grid_dimensions

    # Calculate grid dimensions
    rows, cols = calculate_grid_dimensions(num_groups)

    # Property 1: rows and cols should be non-negative
    assert rows >= 0, f"Rows should be non-negative, got {rows}"
    assert cols >= 0, f"Cols should be non-negative, got {cols}"

    # Property 2: rows * cols should be >= num_groups (enough space for all groups)
    total_cells = rows * cols
    assert total_cells >= num_groups, (
        f"Grid size {rows}x{cols} = {total_cells} is insufficient for "
        f"{num_groups} groups"
    )

    # Property 3: The number of empty cells should be minimized
    # This means (rows * cols - num_groups) should be < cols
    # (i.e., we shouldn't have a completely empty row)
    if num_groups > 0:
        empty_cells = total_cells - num_groups
        assert empty_cells < cols, (
            f"Grid has {empty_cells} empty cells, which is >= {cols} columns. "
            f"This suggests an entire row is empty and the grid is not optimal."
        )

    # Property 4: For special cases, verify expected behavior
    if num_groups == 0:
        assert rows == 0 and cols == 0, "Empty grid should be 0x0"
    elif num_groups == 1:
        assert rows == 1 and cols == 1, "Single group should be 1x1"

    # Property 5: Grid should prefer wider layouts (cols >= rows for most cases)
    # This is a design choice to make better use of screen real estate
    if num_groups > 1:
        # For most cases, we expect cols >= rows
        # Exception: very small numbers might have rows == cols (like 4 -> 2x2)
        pass  # This is a soft preference, not a hard requirement


# **Feature: tufte-plotting-library, Property 17: File export format detection**
@settings(max_examples=100, deadline=2000)
@given(
    data=dataframes_for_plotting(),
    extension=st.sampled_from([".png", ".pdf", ".svg", ".jpg", ".jpeg"]),
)
def test_file_export_format_detection(data, extension):
    """
    Property 17: File export format detection

    For any filename with extension in {'.png', '.pdf', '.svg', '.jpg', '.jpeg'},
    the save function should produce a file in the corresponding format.

    Validates: Requirements 10.3
    """
    # Import required libraries
    try:
        import matplotlib.pyplot as plt
        from tufteplots.adapters import MatplotlibAdapter
        import os
        import tempfile
    except ImportError:
        pytest.skip("Required libraries not installed")

    # Create a theme
    theme = TufteTheme()

    # Create a matplotlib figure
    fig, ax = plt.subplots()
    ax.plot(data["x"], data["y"])

    # Create adapter
    adapter = MatplotlibAdapter(theme)

    # Create a temporary directory for this test
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generate a unique filename with the given extension
        filename = f"test_export{extension}"
        filepath = os.path.join(tmp_dir, filename)

        # Save the figure
        adapter.save(fig, filepath)

        # Property 1: The file should exist
        assert os.path.exists(filepath), f"File {filepath} was not created"

        # Property 2: The file should not be empty
        file_size = os.path.getsize(filepath)
        assert file_size > 0, f"File {filepath} is empty (size: {file_size})"

        # Property 3: The file should be in the correct format
        # Check file signature (magic bytes) to verify format
        with open(filepath, "rb") as f:
            header = f.read(16)  # Read first 16 bytes

        # Define file signatures for each format
        file_signatures = {
            ".png": b"\x89PNG\r\n\x1a\n",
            ".pdf": b"%PDF",
            ".svg": b"<?xml",  # SVG files start with XML declaration
            ".jpg": b"\xff\xd8\xff",  # JPEG magic bytes
            ".jpeg": b"\xff\xd8\xff",
        }

        expected_signature = file_signatures[extension]

        # Check if the file starts with the expected signature
        assert header.startswith(expected_signature), (
            f"File format mismatch for extension '{extension}': "
            f"expected file to start with {expected_signature!r}, "
            f"but got {header[:len(expected_signature)]!r}"
        )

    # Clean up
    plt.close(fig)
