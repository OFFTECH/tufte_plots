"""
Comprehensive demonstration of TuftePlots library capabilities.

This script showcases all major features including:
- Line plots (single and multi-series with direct labels)
- Scatter plots (with and without trend lines)
- Bar plots (with value labels)
- Histograms (with and without rug plots)
- Small multiples
- Theme customization
- Cross-backend comparison (matplotlib, plotly, seaborn)
- File export (PNG, PDF, SVG)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import TuftePlots
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

# Create output directory
OUTPUT_DIR = "examples/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("TuftePlots Comprehensive Demo")
print("=" * 50)


# ============================================================================
# 1. LINE PLOTS
# ============================================================================
print("\n1. Creating line plots...")

# 1.1 Single series line plot
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = np.sin(x) + np.random.normal(0, 0.1, 50)
df_single = pd.DataFrame({"x": x, "y": y})

fig = tufte_line_plot(
    df_single, x="x", y="y", backend="matplotlib", direct_labels=False
)
plt.title("Single Series Line Plot")
plt.savefig(f"{OUTPUT_DIR}/01_line_single.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Single series line plot saved")

# 1.2 Multi-series line plot with direct labels
years = np.arange(2010, 2024)
product_a = 100 + np.cumsum(np.random.normal(5, 2, len(years)))
product_b = 80 + np.cumsum(np.random.normal(7, 3, len(years)))
product_c = 120 + np.cumsum(np.random.normal(4, 2, len(years)))

df_multi = pd.DataFrame(
    {
        "year": np.tile(years, 3),
        "sales": np.concatenate([product_a, product_b, product_c]),
        "product": ["Product A"] * len(years)
        + ["Product B"] * len(years)
        + ["Product C"] * len(years),
    }
)

fig = tufte_line_plot(
    df_multi,
    x="year",
    y="sales",
    hue="product",
    backend="matplotlib",
    direct_labels=True,
)
plt.title("Multi-Series Line Plot with Direct Labels")
plt.xlabel("Year")
plt.ylabel("Sales ($M)")
plt.savefig(
    f"{OUTPUT_DIR}/02_line_multi_direct_labels.png", dpi=150, bbox_inches="tight"
)
plt.close()
print("  ✓ Multi-series line plot with direct labels saved")


# ============================================================================
# 2. SCATTER PLOTS
# ============================================================================
print("\n2. Creating scatter plots...")

# 2.1 Basic scatter plot
np.random.seed(123)
n = 100
df_scatter = pd.DataFrame(
    {"height": np.random.normal(170, 10, n), "weight": np.random.normal(70, 15, n)}
)

fig = tufte_scatter_plot(df_scatter, x="height", y="weight", backend="matplotlib")
plt.title("Height vs Weight")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.savefig(f"{OUTPUT_DIR}/03_scatter_basic.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Basic scatter plot saved")

# 2.2 Scatter plot with trend line
fig = tufte_scatter_plot(
    df_scatter, x="height", y="weight", backend="matplotlib", show_trend=True
)
plt.title("Height vs Weight (with trend line)")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.savefig(f"{OUTPUT_DIR}/04_scatter_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Scatter plot with trend line saved")

# 2.3 Multi-group scatter plot
df_scatter_groups = pd.DataFrame(
    {
        "x": np.random.normal(0, 1, 150),
        "y": np.random.normal(0, 1, 150),
        "group": ["Group A"] * 50 + ["Group B"] * 50 + ["Group C"] * 50,
    }
)
df_scatter_groups["y"] = df_scatter_groups["y"] + df_scatter_groups["group"].map(
    {"Group A": 0, "Group B": 1, "Group C": 2}
)

fig = tufte_scatter_plot(
    df_scatter_groups, x="x", y="y", hue="group", backend="matplotlib"
)
plt.title("Multi-Group Scatter Plot")
plt.savefig(f"{OUTPUT_DIR}/05_scatter_groups.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Multi-group scatter plot saved")


# ============================================================================
# 3. BAR PLOTS
# ============================================================================
print("\n3. Creating bar plots...")

# 3.1 Simple bar plot with value labels
categories = ["Category A", "Category B", "Category C", "Category D", "Category E"]
values = [23, 45, 56, 78, 32]
df_bar = pd.DataFrame({"category": categories, "value": values})

fig = tufte_bar_plot(
    df_bar, x="category", y="value", backend="matplotlib", show_values=True
)
plt.title("Simple Bar Plot with Value Labels")
plt.xlabel("Category")
plt.ylabel("Value")
plt.xticks(rotation=45, ha="right")
plt.savefig(f"{OUTPUT_DIR}/06_bar_simple.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Simple bar plot saved")

# 3.2 Grouped bar plot
quarters = ["Q1", "Q2", "Q3", "Q4"]
df_bar_grouped = pd.DataFrame(
    {
        "quarter": quarters * 2,
        "revenue": [100, 120, 140, 160, 90, 110, 130, 150],
        "year": ["2022"] * 4 + ["2023"] * 4,
    }
)

fig = tufte_bar_plot(
    df_bar_grouped,
    x="quarter",
    y="revenue",
    hue="year",
    backend="matplotlib",
    show_values=False,
)
plt.title("Quarterly Revenue Comparison")
plt.xlabel("Quarter")
plt.ylabel("Revenue ($M)")
plt.savefig(f"{OUTPUT_DIR}/07_bar_grouped.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Grouped bar plot saved")


# ============================================================================
# 4. HISTOGRAMS
# ============================================================================
print("\n4. Creating histograms...")

# 4.1 Basic histogram
np.random.seed(456)
df_hist = pd.DataFrame({"values": np.random.normal(100, 15, 500)})

fig = tufte_histogram(df_hist, column="values", backend="matplotlib", show_rug=False)
plt.title("Distribution of Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig(f"{OUTPUT_DIR}/08_histogram_basic.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Basic histogram saved")

# 4.2 Histogram with rug plot
fig = tufte_histogram(df_hist, column="values", backend="matplotlib", show_rug=True)
plt.title("Distribution with Rug Plot")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig(f"{OUTPUT_DIR}/09_histogram_rug.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Histogram with rug plot saved")


# ============================================================================
# 5. SMALL MULTIPLES
# ============================================================================
print("\n5. Creating small multiples...")

# Create dataset with multiple groups
np.random.seed(789)
regions = ["North", "South", "East", "West"]
months = np.arange(1, 13)

data_list = []
for region in regions:
    base = np.random.uniform(50, 100)
    trend = np.random.uniform(-2, 5)
    noise = np.random.normal(0, 5, len(months))
    sales = base + trend * months + noise

    for month, sale in zip(months, sales):
        data_list.append({"month": month, "sales": sale, "region": region})

df_multiples = pd.DataFrame(data_list)

fig = small_multiples(
    df_multiples,
    x="month",
    y="sales",
    facet_by="region",
    plot_type="line",
    backend="matplotlib",
)
plt.savefig(f"{OUTPUT_DIR}/10_small_multiples.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Small multiples saved")


# ============================================================================
# 6. THEME CUSTOMIZATION
# ============================================================================
print("\n6. Demonstrating theme customization...")

# 6.1 Custom color palette
custom_theme = TufteTheme(
    color_palette=["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"],
    title_size=16,
    label_size=12,
)

theme_manager = ThemeManager(custom_theme)

fig = tufte_line_plot(
    df_multi,
    x="year",
    y="sales",
    hue="product",
    backend="matplotlib",
    direct_labels=True,
    theme=custom_theme,
)
plt.title("Custom Color Palette")
plt.xlabel("Year")
plt.ylabel("Sales ($M)")
plt.savefig(f"{OUTPUT_DIR}/11_custom_colors.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Custom color palette example saved")

# 6.2 Custom font sizes
large_font_theme = TufteTheme(title_size=18, label_size=14, tick_size=11)

fig = tufte_scatter_plot(
    df_scatter, x="height", y="weight", backend="matplotlib", theme=large_font_theme
)
plt.title("Custom Font Sizes")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.savefig(f"{OUTPUT_DIR}/12_custom_fonts.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Custom font sizes example saved")

# 6.3 Theme with grid enabled
grid_theme = TufteTheme(show_grid=True, grid_color="#e0e0e0")

fig = tufte_bar_plot(
    df_bar, x="category", y="value", backend="matplotlib", theme=grid_theme
)
plt.title("Theme with Grid Enabled")
plt.xlabel("Category")
plt.ylabel("Value")
plt.xticks(rotation=45, ha="right")
plt.savefig(f"{OUTPUT_DIR}/13_theme_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Theme with grid example saved")


# ============================================================================
# 7. CROSS-BACKEND COMPARISON
# ============================================================================
print("\n7. Creating cross-backend comparison...")

# Same data, different backends
comparison_data = pd.DataFrame(
    {"x": np.linspace(0, 10, 30), "y": np.sin(np.linspace(0, 10, 30)) * 10 + 50}
)

# Matplotlib version
fig = tufte_line_plot(comparison_data, x="x", y="y", backend="matplotlib")
plt.title("Matplotlib Backend")
plt.savefig(f"{OUTPUT_DIR}/14_backend_matplotlib.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Matplotlib backend example saved")

# Plotly version
try:
    fig = tufte_line_plot(comparison_data, x="x", y="y", backend="plotly")
    fig.update_layout(title="Plotly Backend")
    fig.write_html(f"{OUTPUT_DIR}/15_backend_plotly.html")
    print("  ✓ Plotly backend example saved")
except Exception as e:
    print(f"  ⚠ Plotly backend example skipped: {e}")

# Seaborn version
try:
    fig = tufte_line_plot(comparison_data, x="x", y="y", backend="seaborn")
    plt.title("Seaborn Backend")
    plt.savefig(f"{OUTPUT_DIR}/16_backend_seaborn.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Seaborn backend example saved")
except Exception as e:
    print(f"  ⚠ Seaborn backend example skipped: {e}")


# ============================================================================
# 8. FILE EXPORT EXAMPLES
# ============================================================================
print("\n8. Demonstrating file export formats...")

# Create a sample plot for export
export_data = pd.DataFrame(
    {"category": ["A", "B", "C", "D"], "value": [25, 40, 30, 45]}
)

# PNG export (already done above, but explicitly showing)
fig = tufte_bar_plot(export_data, x="category", y="value", backend="matplotlib")
plt.title("Export Example - PNG")
plt.savefig(f"{OUTPUT_DIR}/17_export_example.png", dpi=150, bbox_inches="tight")
print("  ✓ PNG export saved")

# PDF export
plt.savefig(f"{OUTPUT_DIR}/17_export_example.pdf", bbox_inches="tight")
print("  ✓ PDF export saved")

# SVG export
plt.savefig(f"{OUTPUT_DIR}/17_export_example.svg", bbox_inches="tight")
print("  ✓ SVG export saved")

plt.close()


# ============================================================================
# 9. APPLYING TUFTE STYLE TO EXISTING FIGURES
# ============================================================================
print("\n9. Applying Tufte style to existing figures...")

# Create a standard matplotlib figure
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(comparison_data["x"], comparison_data["y"], linewidth=2)
ax.set_title("Before Tufte Style")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True)
plt.savefig(f"{OUTPUT_DIR}/18_before_tufte.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Before Tufte style saved")

# Apply Tufte style to the same plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(comparison_data["x"], comparison_data["y"], linewidth=2)
ax.set_title("After Tufte Style")
ax.set_xlabel("X")
ax.set_ylabel("Y")
fig = apply_tufte_style(fig, backend="matplotlib")
plt.savefig(f"{OUTPUT_DIR}/19_after_tufte.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ After Tufte style saved")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 50)
print("Demo complete!")
print(f"All plots saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
print("  - Line plots: 01-02")
print("  - Scatter plots: 03-05")
print("  - Bar plots: 06-07")
print("  - Histograms: 08-09")
print("  - Small multiples: 10")
print("  - Theme customization: 11-13")
print("  - Backend comparison: 14-16")
print("  - Export formats: 17 (PNG, PDF, SVG)")
print("  - Before/After Tufte: 18-19")
print("=" * 50)
