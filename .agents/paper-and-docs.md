# Paper and Documentation System

## Paper (`content/paper/`)

The research paper "Structural Estimation of Life Cycle Models with Wealth in
the Utility Function" is written in **MyST Markdown** and built with the
[MyST](https://mystmd.org/) toolchain.

### Key files

| File | Purpose |
|------|---------|
| `content/paper/01-paper.md` | Main paper source (MyST Markdown with directives, cross-refs, citations) |
| `content/paper/main.bib` | BibTeX bibliography |
| `content/paper/math.ipynb` | Notebook with mathematical derivations |
| `content/paper/images/` | Static images referenced in the paper |
| `content/paper/structural_estimation_pdf_tex/` | LaTeX/PDF output from paper build |

### Building the paper

```bash
myst build          # Build the MyST site (HTML)
myst build --pdf    # Build PDF output
```

The GitHub Actions workflow `deploy.yml` builds and deploys to GitHub Pages on
push to `main`.

### MyST configuration (`myst.yml`)

The `myst.yml` file at the repo root is ~800 lines, primarily because it
contains an extensive set of LaTeX math macros shared across the paper. These
macros follow Econ-ARK notation conventions:

```yaml
math:
  \CRRA: \rho
  \DiscFac: \beta
  \Rfree: \mathsf{R}
  # ... hundreds more
```

When editing the paper, use these macros rather than raw LaTeX symbols to
maintain consistency with the broader Econ-ARK ecosystem.

### Citation format

Uses `sphinxcontrib-bibtex` style references:
```markdown
{cite}`carroll2004` or {cite:t}`carroll2004`
```

## Sphinx documentation (`docs/`)

A separate Sphinx-based documentation site exists in `docs/`:
- `docs/conf.py` — Sphinx config (myst_parser, autodoc, Furo theme)
- `docs/index.md` — Landing page (includes README)
- Built for ReadTheDocs (`/.readthedocs.yaml`)

Build locally:
```bash
nox -s docs
```

## Slides (`content/slides/`)

Presentation slides built with Quarto:
- `slides.ipynb` — Source notebook
- `slides.tex` — LaTeX output
- `strucutral_estimation.html` — HTML output (note: typo in filename is
  intentional/historical)

## Generated content

Estimation runs produce artifacts checked into version control:

### Tables (`content/tables/`)

- `min/` — Minimum-distance estimates (e.g. `IndShock_estimate_results.csv`)
- `msm/` — Method-of-simulated-moments estimates
- `TRP/` — Target Retirement Portfolio estimates

Format: headerless two-column CSV (`parameter_name, value`), plus metadata rows
like `time_to_estimate`.

### Figures (`content/figures/`)

SVG, PNG, and PDF figures:
- `*Sensitivity.svg` — Sensitivity bar charts
- `*SMMcontour.svg` — Contour plots of the objective function

### Notebooks (`src/notebooks/`)

Jupyter notebooks for individual model estimations and analysis:
- `IndShock.ipynb`, `Portfolio.ipynb`, `WarmGlow.ipynb`, etc.
- `Model_Comparisons.ipynb` — Cross-model comparison
- `SCF_notebook.ipynb` — SCF data exploration
- `parse_tables.ipynb` — Table formatting utilities

### MSM notebooks (`src/msm_notebooks/`)

Separate notebooks for method-of-simulated-moments estimation, including
covariance computation notebooks (`NetWorth_Cov.ipynb`, `FinAssets_Cov.ipynb`).

## Table of Contents (`_toc.yml`)

Defines the MyST site structure:
```yaml
root: README
chapters:
  - file: content/paper/01-paper
```
