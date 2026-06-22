import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')

TARGET_IDS = {'4195616c', '0b397e2a', '3bef81ae'}

nb = json.load(open('comparison_analysis.ipynb', encoding='utf-8'))
patched = 0

for cell in nb['cells']:
    if cell.get('id') not in TARGET_IDS:
        continue

    new_source = []
    for line in cell['source']:
        stripped = line.strip()

        # reemplaza FIG_DIR = Path(...)
        if re.search(r'FIG_DIR\s*=\s*Path\(', line):
            new_source.append('from pathlib import Path\n')
            new_source.append('import os as _os\n')
            new_source.append(
                'FIG_DIR = Path(_os.path.abspath("comparison_analysis.ipynb"))'
                '.parent.parent / "resultados_metricas" / "figures"\n'
            )
            continue

        # actualiza mkdir
        if 'FIG_DIR.mkdir' in line:
            new_source.append('FIG_DIR.mkdir(parents=True, exist_ok=True)\n')
            continue

        # elimina todo lo relacionado con tex_fig_dir y shutil.copy a resultados
        if 'tex_fig_dir' in stripped:
            continue
        if 'auto-copy' in stripped:
            continue
        if 'shutil.copy' in stripped and 'resultados_metricas' in stripped:
            continue
        # elimina shutil.copy donde el destino use tex_fig_dir (aunque no diga resultados)
        if stripped.startswith('shutil.copy') and 'tex_fig_dir' in stripped:
            continue

        new_source.append(line)

    cell['source'] = new_source
    patched += 1

with open('comparison_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Parcheadas {patched} celdas")
