# Figure QA notes

## Figure contracts

### `expanded_training_curves`

- Core conclusion: the expanded-data PaiNN fit reduces training error, while validation errors improve early and then fluctuate around a higher plateau.
- Evidence roles: panel a shows energy learning; panel b independently checks the coordinate derivative target (forces).
- Archetype: quantitative grid.
- Source data: `results/expanded_training_history.csv`.
- All 30 epochs come from the persisted JSON log. The best checkpoint by validation force RMSE occurred at epoch 23 and is marked in panel b.

### `fixed_test_comparison`

- Core conclusion: broader reaction coverage improves generalization on the same held-out formulas and reactions; increasing the diverse training set further lowers all three errors in this run.
- Evidence roles: absolute energy, reaction-relative energy, and forces test distinct MLIP outputs with different physical interpretations.
- Archetype: quantitative comparison grid.
- Source data: `results/fixed_test_comparison.csv`.
- The y axes are logarithmic and use plain decimal tick labels. Lower values are better.

## Experimental scope

- All checkpoints are evaluated on the same test split: 150 configurations, 8 formulas, and 15 reactions.
- One random seed (`7`) was used. No error bars are shown because repeated-seed uncertainty was not measured.
- These are compact-reproduction results, not a reproduction of the full Transition1x paper benchmark.

## Automated and visual QA

- Python source preflight: pass (21 checks, no warnings or failures).
- Panel alignment: pass at 1.5 pt tolerance for both figures.
- PDF glyph floor: pass; minimum detected sizes are 6.2 pt and 6.1 pt.
- Rendered collision audit: pass; no failures or warnings for either figure.
- Final-size visual inspection: labels, legends, panel hierarchy, missing-epoch gap, log axes, and bar annotations inspected.
- Exports: editable SVG, PDF, 300 dpi PNG preview, and 600 dpi TIFF.

The visual style is an original project implementation informed by the layout and palette grammar of `figures4papers`; no upstream manuscript values, labels, or assets are reused.
