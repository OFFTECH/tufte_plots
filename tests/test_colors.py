import pandas as pd
import numpy as np
import tufteplots as tp
import matplotlib.pyplot as plt

print("Initializing example plot generation...")
print("Theme Font:", tp.TufteTheme().font_family)
print("Theme Colors:", tp.TufteTheme().color_palette)

np.random.seed(42)

# 1. Line Plot
df_line = pd.DataFrame(
    {
        "Time": range(20),
        "Series A": np.random.randn(20).cumsum(),
        "Series B": np.random.randn(20).cumsum() + 5,
    }
)
df_line_melted = df_line.melt("Time", var_name="Series", value_name="Value")
fig1 = tp.tufte_line_plot(
    df_line_melted, "Time", "Value", hue="Series", backend="matplotlib"
)
fig1.savefig("example_line_plot.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Generated example_line_plot.png")

# 2. Scatter Plot
df_scatter = pd.DataFrame(
    {
        "Group A X": np.random.randn(50),
        "Group A Y": np.random.randn(50),
        "Group B X": np.random.randn(50) + 2,
        "Group B Y": np.random.randn(50) + 2,
    }
)
df_scatter_melt = pd.DataFrame(
    {
        "X": np.concatenate([df_scatter["Group A X"], df_scatter["Group B X"]]),
        "Y": np.concatenate([df_scatter["Group A Y"], df_scatter["Group B Y"]]),
        "Category": ["Group A"] * 50 + ["Group B"] * 50,
    }
)
fig2 = tp.tufte_scatter_plot(
    df_scatter_melt, "X", "Y", hue="Category", show_trend=True, backend="matplotlib"
)
fig2.savefig("example_scatter_plot.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Generated example_scatter_plot.png")

# 3. Bar Plot
df_bar = pd.DataFrame({"Category": ["A", "B", "C", "D"], "Value": [15, 30, 45, 20]})
fig3 = tp.tufte_bar_plot(df_bar, "Category", "Value", backend="matplotlib")
fig3.savefig("example_bar_plot.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Generated example_bar_plot.png")

# 4. Histogram
df_hist = pd.DataFrame({"Values": np.random.randn(500) * 10 + 50})
fig4 = tp.tufte_histogram(
    df_hist, "Values", bins=20, show_rug=True, backend="matplotlib"
)
fig4.savefig("example_histogram.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print("Generated example_histogram.png")

print("All example plots successfully generated.")
