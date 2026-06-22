import json, sys
sys.stdout.reconfigure(encoding='utf-8')

new_source = """\
# ===========================================================
# LORE: Fidelidad del árbol surrogate al NN
#
# tree_fidelity_lore_local  → % instancias donde el árbol replica bb_local
# tree_fidelity_lore_global → % instancias donde el árbol replica bb_global
# ===========================================================

ORDER = [
    'IID Balanced IID',
    'Gaussian Affine σ=0.05',
    'Gaussian Affine σ=0.3',
    'Dirichlet (label skew) α=0.5',
    'Dirichlet (label skew) α=0.7',
]

FID_KEYS = ['tree_fidelity_lore_local', 'tree_fidelity_lore_global']
datasets  = sorted(df_all['dataset'].unique())

short = lambda s: (s.replace('IID Balanced ', '')
                    .replace('Gaussian Affine ', 'Gauss ')
                    .replace('Dirichlet (label skew) ', 'Dir '))

COLOR_LOCAL  = '#EA580C'  # colorLoreLocal
COLOR_GLOBAL = '#8B5CF6'  # colorLoreGlobal

# ── Tabla ──────────────────────────────────────────────────
print("=" * 70)
print("  LORE: FIDELIDAD DEL ÁRBOL SURROGATE  (media sobre 4/6/8 clientes)")
print("=" * 70)
for dataset in datasets:
    sub   = df_all[df_all['dataset'] == dataset]
    avail = [k for k in FID_KEYS if k in sub.columns]
    tabla = (
        sub.groupby('label')[avail].mean()
           .reindex([o for o in ORDER if o in sub['label'].unique()])
           .rename(columns={
               'tree_fidelity_lore_local':  'Fidelidad bb_local',
               'tree_fidelity_lore_global': 'Fidelidad bb_global',
           })
    )
    print(f"\\n  {dataset}")
    print(tabla.to_string(float_format='{:.3f}'.format))

# ── Gráfica por dataset ────────────────────────────────────
for dataset in datasets:
    sub   = df_all[df_all['dataset'] == dataset]
    avail = [k for k in FID_KEYS if k in sub.columns]
    orden = [o for o in ORDER if o in sub['label'].unique()]
    t     = sub.groupby('label')[avail].mean().reindex(orden)

    fig, ax = plt.subplots(figsize=(7, 5))
    x     = np.arange(len(t))
    width = 0.35

    bars0 = ax.bar(x - width/2, t[avail[0]], width,
                   color=COLOR_LOCAL, alpha=0.9, edgecolor='white', label='Fidelidad bb_local')
    bars1 = ax.bar(x + width/2, t[avail[1]], width,
                   color=COLOR_GLOBAL, alpha=0.9, edgecolor='white', label='Fidelidad bb_global')

    ax.set_xticks(x)
    ax.set_xticklabels([short(o) for o in orden], rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.set_title(f'LORE — Fidelidad del árbol surrogate al NN — {dataset}', fontsize=11)
    ax.set_ylabel('Fidelidad al NN (replica predicciones)')
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')

    for bar in list(bars0) + list(bars1):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fname = f'lore_fidelity_{dataset}.png'
    plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Guardado: {fname}")\
"""

nb = json.load(open('comparison_analysis.ipynb', encoding='utf-8'))
patched = False
for cell in nb['cells']:
    if cell.get('id') == '4f8013fc':
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
