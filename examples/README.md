# TuftePlots Examples

This directory contains comprehensive examples demonstrating the capabilities of the TuftePlots library.

## Running the Examples

To run the comprehensive demo:

```bash
python examples/comprehensive_demo.py
```

This will generate various plot types and save them to the `examples/output/` directory.

## Requirements

Make sure you have TuftePlots installed with all dependencies:

```bash
pip install -e .
```

## What's Included

The `comprehensive_demo.py` script demonstrates:

1. **Line Plots**
   - Single series line plots
   - Multi-series line plots with direct labels
   - Comparison across backends (matplotlib, plotly, seaborn)

2. **Scatter Plots**
   - Basic scatter plots
   - Scatter plots with trend lines
   - Multi-group scatter plots

3. **Bar Plots**
   - Simple bar plots with value labels
   - Grouped bar plots

4. **Histograms**
   - Basic histograms
   - Histograms with rug plots

5. **Small Multiples**
   - Grid layouts for comparing related datasets
   - Shared axes across subplots

6. **Theme Customization**
   - Custom color palettes
   - Custom font sizes
   - Feature toggles (grid, range frames, etc.)

7. **Cross-Backend Comparison**
   - Same data plotted with matplotlib, plotly, and seaborn
   - Visual consistency verification

8. **File Export**
   - PNG, PDF, and SVG export examples
   - Font embedding for vector formats

## Output

All generated plots are saved to `examples/output/` with descriptive filenames.
