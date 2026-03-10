# Build System and CI

## Package build

**Build backend**: Hatchling with hatch-vcs for version management.

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"
```

Version is derived from git tags and written to `src/estimark/_version.py`.
Build with:

```bash
nox -s build    # or: python -m build
```

## Dependency management

Core dependencies are intentionally empty in `[project.dependencies]` — all
runtime deps are in optional groups:

| Group  | Deps                                                           | Install               |
| ------ | -------------------------------------------------------------- | --------------------- |
| `run`  | numpy, pandas, matplotlib, scipy, statsmodels, estimagic, HARK | `pip install .[run]`  |
| `test` | pytest, pytest-cov                                             | `pip install .[test]` |
| `dev`  | pytest, pytest-cov                                             | `pip install .[dev]`  |
| `docs` | sphinx, myst_parser, furo, etc.                                | `pip install .[docs]` |

HARK is installed from GitHub main:
`HARK @ git+https://github.com/econ-ark/HARK@main`

A conda environment file also exists: `environment.yml` (Python 3.12, includes
Dask for parallel MSM estimation).

## Nox sessions (`noxfile.py`)

| Session          | Command                 | Purpose                                                |
| ---------------- | ----------------------- | ------------------------------------------------------ |
| `lint`           | `nox -s lint`           | Run pre-commit on all files                            |
| `pylint`         | `nox -s pylint`         | Run PyLint on `estimark`                               |
| `tests`          | `nox -s tests`          | Run pytest                                             |
| `docs`           | `nox -s docs`           | Build Sphinx docs (with autobuild in interactive mode) |
| `build_api_docs` | `nox -s build_api_docs` | Generate API docs with sphinx-apidoc                   |
| `build`          | `nox -s build`          | Build sdist + wheel                                    |

Default sessions: `lint`, `pylint`, `tests`.

## Pre-commit hooks (`.pre-commit-config.yaml`)

| Hook               | Purpose                                                                   |
| ------------------ | ------------------------------------------------------------------------- |
| blacken-docs       | Format Python in docs                                                     |
| pre-commit-hooks   | Standard checks (large files, merge conflicts, trailing whitespace, etc.) |
| pygrep-hooks       | RST syntax checks                                                         |
| prettier           | Format YAML, Markdown, HTML, CSS, JS, JSON (with `--prose-wrap=always`)   |
| ruff               | Python linting + formatting                                               |
| mypy               | Type checking (on `src/` and `tests/`)                                    |
| codespell          | Spell checking                                                            |
| shellcheck         | Shell script linting                                                      |
| validate-pyproject | Validate pyproject.toml                                                   |
| check-jsonschema   | Validate dependabot, GitHub workflows, ReadTheDocs configs                |

**Important**: Pre-commit excludes `content/tables/` CSVs and
`.copier-answers.yml`.

## CI workflows (`.github/workflows/`)

### `ci.yml` — Main CI

Triggered on push to `main` and PRs.

1. **Format job**: pre-commit + PyLint
2. **Checks job**: pytest across matrix:
   - Python: 3.8, 3.13, pypy-3.10
   - OS: ubuntu-latest, windows-latest, macos-14
   - Coverage uploaded to Codecov

### `cd.yml` — Continuous Deployment

Triggered on releases. Builds sdist/wheel and publishes to Test PyPI.

### `deploy.yml` — Documentation Deployment

Triggered on push to `main`. Builds MyST site and deploys to GitHub Pages.

## Linting and formatting configuration

All in `pyproject.toml`:

```toml
[tool.ruff]
src = ["src"]
# ...

[tool.mypy]
files = "src"
python_version = "3.8"
strict = true
# Various overrides for untyped HARK imports

[tool.pytest.ini_options]
testpaths = ["tests"]
```

## Template origin

This repo was generated from the
[scientific-python/cookie](https://github.com/scientific-python/cookie) template
(see `.copier-answers.yml`). Many config files follow that template's
conventions.

## Known issues

- `src/tests.py` has a broken import (`from estimark.min import estimate_min`);
  the correct module is `estimark.estimation`. This file is not run by pytest
  (which only looks in `tests/`).
- `n_cores: 12` is hardcoded in `parameters.py` minimize_options — may need
  adjustment for different machines.
- The actual test suite (`tests/test_package.py`) only checks the package
  version; there are no functional tests for the estimation pipeline.
