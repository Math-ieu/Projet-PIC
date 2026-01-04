import json

notebook_path = "notebooks/02_baseline_cnn.ipynb"

with open(notebook_path, 'r') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "IMG_SIZE = (256, 256)" in source and "IMG_HEIGHT" not in source:
            # Inject definitions
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if "IMG_SIZE = (256, 256)" in line:
                    new_source.append("IMG_HEIGHT, IMG_WIDTH = IMG_SIZE\n")
            cell['source'] = new_source
            found = True
            break

if found:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=4)
    print(f"Updated {notebook_path}")
else:
    print(f"Could not find target cell or IMG_HEIGHT already defined in {notebook_path}")
