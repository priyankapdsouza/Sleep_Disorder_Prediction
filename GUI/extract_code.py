import re
import sys

if len(sys.argv) < 3:
    print("Usage: python extract_code.py <notebook_path> <output_py_path>")
    sys.exit(1)

notebook_path = sys.argv[1]
output_path = sys.argv[2]

# Read the notebook file as text
with open(notebook_path, 'r') as f:
    content = f.read()

# Find all code cells using regex
code_cells = re.findall(r'"cell_type": "code".*?"source": \[(.*?)\]', content, re.DOTALL)

# Extract the code from each cell
all_code = []
for cell in code_cells:
    # Remove quotes and join lines
    lines = re.findall(r'"([^"]*)"', cell)
    code = '\n'.join(lines)
    all_code.append(code)

# Write to a Python script
with open(output_path, 'w') as f:
    for code in all_code:
        f.write(code + '\n\n')
