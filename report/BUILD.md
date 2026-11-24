# Build the English report

Requirements:

- a TeX distribution containing XeLaTeX and `latexmk`;
- `siunitx`, `natbib`, and standard scientific LaTeX packages;
- the two PDF figures in `../figures/`.

From the repository root:

```bash
cd report
make
```

The compiled report is `report/main.pdf`. Rebuild the PDF after editing `main.tex`.
The generated auxiliary files are ignored; the reviewed PDF may be committed as a
convenient project artifact.
