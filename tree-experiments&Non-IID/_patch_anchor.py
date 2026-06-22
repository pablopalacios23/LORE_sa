import json, sys
sys.stdout.reconfigure(encoding='utf-8')

new_source = """\
# ===========================================================
# Anchor: precisión, cobertura y disponibilidad — grouped bar
# Figura 1 (por dataset): Precisión + Cobertura (2 subplots)
# Figura 2 (por dataset): Disponibilidad (1 subplot)
# ===========================================================

ORDER = [
    'IID Balanced IID',
    'Gaussian Affine σ=0.05',
    'Gaussian Affine σ=0.3',
    'Dirichlet (label skew) α=0.5',
    'Dirichlet (label skew) α=0.7'
]

PREC_KEYS  = ['anchor_prec_testlocal_local', 'anchor_prec_testlocal_global',
              'anchor_prec_testglobal_local', 'anchor_prec_testglobal_global']
COV_KEYS   = ['anchor_cov_testlocal_local',  'anchor_cov_testlocal_global',
              'anchor_cov_testglobal_local',  'anchor_cov_testglobal_global']
RATIO_KEYS = ['ratio_has_anchor_testlocal_local',  'ratio_has_anchor_testlocal_global',
              'ratio_has_anchor_testglobal_local',  'ratio_has_anchor_testglobal_global']

BAR_LABELS = ['bb_local / test=local', 'bb_global / test=local',
              'bb_local / test=global', 'bb_global / test=global']
COL_COLORS = ['#1F77B4', '#D62728', '#2CA02C', '#9467BD']

datasets = sorted(df_all['dataset'].unique())

short = lambda s: (s.replace('IID Balanced ', '')
                    .replace('Gaussian Affine ', 'Gauss ')
                    .replace('Dirichlet (label skew) ', 'Dir '))

# ── Tablas ─────────────────────────────────────────────────
for keys, title in [(PREC_KEYS, 'PRECISIÓN'), (COV_KEYS, 'COBERTURA'), (RATIO_KEYS, 'DISPONIBILIDAD')]:
    for dataset in datasets:
        sub   = df_all[df_all['dataset'] == dataset]
        orden = [o for o in ORDER if o in sub['label'].unique()]
        t = sub.groupby('label')[keys].mean().reindex(orden)
        t.columns = BAR_LABELS
        print(f"\\n{'='*72}\\n  Anchor {title} — {dataset}\\n{'='*72}")
        print(t.to_string(float_format='{:.3f}'.format))

def draw_bars(ax, tabla, keys, ylabel, title):
    orden  = list(tabla.index)
    x      = np.arange(len(orden))
    n      = len(keys)
    width  = 0.18
    offset = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    for i, (key, label, color) in enumerate(zip(keys, BAR_LABELS, COL_COLORS)):
        vals = tabla[key].values
        bars = ax.bar(x + offset[i], vals, width, color=color, alpha=0.9,
                      edgecolor='white', label=label)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([short(o) for o in orden], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', ncol=2)

# ── Figura 1: Precisión + Cobertura por dataset ────────────
for dataset in datasets:
    sub   = df_all[df_all['dataset'] == dataset]
    orden = [o for o in ORDER if o in sub['label'].unique()]
    tabla = sub.groupby('label')[PREC_KEYS + COV_KEYS].mean().reindex(orden)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9))

    draw_bars(axes[0], tabla, PREC_KEYS, 'Precisión (0–1)', f'Anchor — Precisión — {dataset}')
    draw_bars(axes[1], tabla, COV_KEYS,  'Cobertura (0–1)', f'Anchor — Cobertura — {dataset}')

    plt.tight_layout()
    fname = f'anchor_prec_cov_{dataset}.png'
    plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Guardado: {fname}")

# ── Figura 2: Disponibilidad por dataset ───────────────────
for dataset in datasets:
    sub   = df_all[df_all['dataset'] == dataset]
    orden = [o for o in ORDER if o in sub['label'].unique()]
    tabla = sub.groupby('label')[RATIO_KEYS].mean().reindex(orden)

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_bars(ax, tabla, RATIO_KEYS, 'Disponibilidad (0–1)', f'Anchor — Disponibilidad — {dataset}')

    plt.tight_layout()
    fname = f'anchor_availability_{dataset}.png'
    plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Guardado: {fname}")\
"""

nb = json.load(open('comparison_analysis.ipynb', encoding='utf-8'))
patched = False
for cell in nb['cells']:
    if cell.get('id') == '23150bf2':
        cell['source'] = [line + '\n' for line in new_source.splitlines()]
        cell['source'][-1] = cell['source'][-1].rstrip('\n')
        patched = True
        break

if not patched:
    print("ERROR: celda no encontrada")
else:
    with open('comparison_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Done")
