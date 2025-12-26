#idk i encountered some metadata bug, should be fixed by clearing it - it makes it so that colab can't open the notebook
# and github can't display the notebook properly

import nbformat
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
notebook_path = os.path.join(script_dir, "gnn_attempt.ipynb")
output_path = os.path.join(script_dir, "gnn_attempt_fixed.ipynb")

print(f"Reading notebook from: {notebook_path}")

# Read the notebook with the correct version parameter
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Remove the problematic widgets metadata
if "widgets" in nb.metadata:
    print("Removing 'widgets' from metadata...")
    del nb.metadata["widgets"]

print(f"Writing fixed notebook to: {output_path}")
with open(output_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Done! Notebook fixed successfully.")
