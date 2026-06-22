import json, sys
sys.stdout.reconfigure(encoding='utf-8')

new_source = """\
# ===========================================================
# LIME: ¿Asignan el modelo local y el global la misma
#       importancia a las mismas features?
# Métrica: delta_lime_testglobal / delta_lime_testlocal
#   → diferencia media (L1) entre los vectores de importancia
#     del modelo local y del global, normalizados a L1=1
#   → 0 = idénticos; 1 = completamente distintos
# ===========================================================

ORDER = [
    'IID Balanced IID',
    'Gaussian Affine σ=0.05',
    'Gaussian Affine σ=0.3',
    'Dirichlet (label skew) α=0.5',
    'Dirichlet (label skew) α=0.7',
]

datasets = sorted(df_all['dataset'].unique())

COLOR_LOCAL  = '#FF7F0E'  # colorAgreementLocal  → Local Test
COLOR_GLOBAL = '#17BECF'  # colorAgreementGlobal → Global Test

short = lambda s: (s.replace('IID Balanced ', '')
                    .replace('Gaussian Affine ', 'Gauss ')
                    .replace('Dirichlet (label skew) ', 'Dir '))

# ── Tabla ──────────────────────────────────────────────────
print("=" * 70)
print("  LIME: DELTA — diferencia media de importancias local vs global")
print("  (media sobre 4/6/8 clientes)")
print("=" * 70)
for dataset in datasets:
    sub   = df_all[df_all['dataset'] == dataset]
    tabla = (
        sub.groupby('label')[['delta_lime_testglobal', 'delta_lime_testlocal']]
           .mean()
           .rename(columns={'delta_lime_testglobal': 'Global Test',
                            'delta_lime_testlocal':  'Local Test'})
    )
    tabla = tabla.reindex([o for o in ORDER if o in tabla.index])
    tabla['Diff (G-L)'] = tabla['Global Test'] - tabla['Local Test']
    print(f"\\n  {dataset}")
    print(tabla.to_string(float_format='{:.4f}'.format))

# ── Gráfica por dataset ────────────────────────────────────
for dataset in datasets:
    sub   = df_all[df_all['dataset'] == dataset]
    tabla = (
        sub.groupby('label')[['delta_lime_testglobal', 'delta_lime_testlocal']]
           .mean()
           .reindex([o for o in ORDER if o in sub['label'].unique()])
    )
    x     = np.arange(len(tabla))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))

    bars_g = ax.bar(x - width/2, tabla['delta_lime_testglobal'],
                    width, color=COLOR_GLOBAL, alpha=0.9, edgecolor='white',
                    label='Global Test')
    bars_l = ax.bar(x + width/2, tabla['delta_lime_testlocal'],
                    width, color=COLOR_LOCAL,  alpha=0.9, edgecolor='white',
                    label='Local Test')

    ax.set_xticks(x)
    ax.set_xticklabels([short(o) for o in tabla.index], rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0.0, 0.25)
    ax.set_title(f'LIME: Δ importancia local vs global — {dataset}', fontsize=11)
    ax.set_ylabel('Δ LIME (0 = idéntico, 1 = opuesto)')
    ax.grid(axis='y', alpha=0.3)

    for bar in list(bars_g) + list(bars_l):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()

    fname = f'lime_feature_importance_{dataset}.png'
    plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Guardado: {fname}")\
"""

nb = json.load(open('comparison_analysis.ipynb', encoding='utf-8'))
patched = False
for cell in nb['cells']:
    if cell.get('id') == '56abc354':
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
