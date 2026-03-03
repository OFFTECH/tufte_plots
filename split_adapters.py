import ast
import os
import shutil

source_file = "tufteplots/adapters.py"
adapters_dir = "tufteplots/adapters"

if not os.path.exists(adapters_dir):
    os.makedirs(adapters_dir)

with open(source_file, "r", encoding="utf-8") as f:
    source = f.read()

lines = source.split("\n")
tree = ast.parse(source)

nodes = {}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        nodes[node.name] = (node.lineno - 1, node.end_lineno)

# Find the last import line to get the header
header_end = 0
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign)):
        header_end = node.end_lineno
    elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        break

header = "\n".join(lines[:header_end]) + "\n\n"


def get_source(name):
    start, end = nodes[name]
    # include decorators if any, but start already handles it in some cases.
    # actually node.lineno for class/def might be after decorators.
    # ast node has .decorator_list, let's just use start-end.
    # Wait, node.lineno is the 'def' line, so we need to account for decorators.
    # Let's adjust start based on decorator_list
    node_obj = [n for n in tree.body if getattr(n, "name", "") == name][0]
    if hasattr(node_obj, "decorator_list") and node_obj.decorator_list:
        real_start = node_obj.decorator_list[0].lineno - 1
    else:
        real_start = start
    return "\n".join(lines[real_start:end]) + "\n"


# 1. base.py
with open(f"{adapters_dir}/base.py", "w", encoding="utf-8") as f:
    f.write(header)
    f.write(get_source("BackendAdapter"))

# 2. grid.py
with open(f"{adapters_dir}/grid.py", "w", encoding="utf-8") as f:
    f.write(header)
    f.write("from .base import BackendAdapter, FigureType\n\n")
    f.write(get_source("calculate_grid_dimensions") + "\n")
    f.write(get_source("small_multiples") + "\n")
    f.write(get_source("_create_matplotlib_small_multiples") + "\n")
    f.write(get_source("_create_plotly_small_multiples") + "\n")

# 3. matplotlib_adapter.py
with open(f"{adapters_dir}/matplotlib_adapter.py", "w", encoding="utf-8") as f:
    f.write(header)
    f.write("from .base import BackendAdapter, FigureType\n\n")
    f.write(get_source("MatplotlibAdapter"))

# 4. plotly_adapter.py
with open(f"{adapters_dir}/plotly_adapter.py", "w", encoding="utf-8") as f:
    f.write(header)
    f.write("from .base import BackendAdapter, FigureType\n\n")
    f.write(get_source("PlotlyAdapter"))

# 5. seaborn_adapter.py
with open(f"{adapters_dir}/seaborn_adapter.py", "w", encoding="utf-8") as f:
    f.write(header)
    f.write("from .base import BackendAdapter, FigureType\n\n")
    f.write(get_source("SeabornAdapter"))

# 6. __init__.py
with open(f"{adapters_dir}/__init__.py", "w", encoding="utf-8") as f:
    f.write('"""Backend adapters for TuftePlots."""\n\n')
    f.write("from .base import BackendAdapter\n")
    f.write("from .matplotlib_adapter import MatplotlibAdapter\n")
    f.write("from .plotly_adapter import PlotlyAdapter\n")
    f.write("from .seaborn_adapter import SeabornAdapter\n")
    f.write("from .grid import small_multiples, calculate_grid_dimensions\n\n")
    f.write("__all__ = [\n")
    f.write('    "BackendAdapter",\n')
    f.write('    "MatplotlibAdapter",\n')
    f.write('    "PlotlyAdapter",\n')
    f.write('    "SeabornAdapter",\n')
    f.write('    "small_multiples",\n')
    f.write('    "calculate_grid_dimensions",\n')
    f.write("]\n")

print("Splitting complete.")
