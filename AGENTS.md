# Repository Guidelines

## Project Structure & Module Organization
- Source: place Python packages under `src/` (e.g., `src/bqgnn/`).
- Tests: keep unit tests in `tests/` mirroring package paths (`tests/test_*.py`).
- Scripts & experiments: use `scripts/` and `notebooks/` for runnable demos and research.
- Data & models: never commit large artifacts; use `data/` and `models/` in `.gitignore`.

Example tree:
```
src/bqgnn/
tests/
scripts/
notebooks/
```

## Build, Test, and Development Commands
- Setup (editable install + dev deps): `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]`
- Run tests (quiet): `pytest -q`
- Lint & format: `ruff check . && ruff format .` (or `black .` if preferred)
- Run example script: `python -m bqgnn.cli --help`

If a `Makefile` exists, prefer: `make init`, `make test`, `make lint`.

## Coding Style & Naming Conventions
- Indentation: 4 spaces; max line length 100–120 chars.
- Naming: packages `snake_case`, classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_SNAKE_CASE`.
- Imports: standard library, third‑party, local (grouped with blank lines).
- Type hints: use Python typing throughout; run `mypy` if configured.

## Testing Guidelines
- Framework: `pytest` with `pytest-cov` for coverage.
- Structure: mirror `src/` and name tests `test_*.py`.
- Expectations: add tests for new features/bugfixes; aim for ≥80% coverage of touched code.
- Run focused tests: `pytest tests/path/test_file.py::test_case`.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (e.g., `feat: add graph encoder`, `fix: correct GNN layer dims`).
- Scope small and atomic; include rationale in the body when helpful.
- PRs: include description, motivation, screenshots/CLI output, and reproduction steps; link issues (`Closes #123`).
- Checks: ensure tests pass and linters are clean before requesting review.

## Security & Configuration Tips
- Do not commit secrets; use `.env` and provide `.env.example`.
- Large data/models: store externally; document how to fetch or generate.
- Reproducibility: pin dependencies in `pyproject.toml`/`requirements.txt` and set random seeds in experiments.



# Guideline for Research Engineering Tasks
This guideline is for a research engineering task that follows **Explore → Plan → Code → Test → Write-up**. It incorporates an active, self-updating to-do list, strict determinism policy, and a clean `work_log/<run_id>/` layout.

---

## 0. Summary

Goal. Ship a correct, reproducible, and well documented change suitable for research workflows.  
Acceptance criteria. CI green, numerical equivalence within tolerance on all declared devices, complete docs and changelog, exact steps to reproduce with fixed seeds and recorded environment, and artifacts stored under `work_log/<run_id>/`.  
Risk policy. If evidence contradicts any assumption, stop and revise the plan before coding.

---

## 1) Explore

Record concrete evidence from the codebase, not impressions.

### 1.1 Targets, examples, dependencies

| Path | Role | Entry points | Notes |
|---|---|---|---|
| fill in | edit target | functions or classes | link docstrings |
| fill in | comparable example | functions or classes | link to tests |
| fill in | dependency | imported modules | versions and pins |

### 1.2 API checklist (read defining files)

| Module path | Symbol | Full signature | Required args | Optional args + defaults | Return type/shape | Device/dtype assumptions | RNG needs | Error modes | Side effects | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| fill in | fill in | `def name(args) -> type` | list | list | type, shape | cpu, cuda, mps | seed, generator | exceptions | I/O, global state | links |

Rule. Do not proceed until every API used is verified against its defining file and tests.

### 1.3 Static verification

Summarize what you read: docstrings, type hints, wrappers, deprecations, guards, data movement, dtype casts. Paste file paths and line ranges.

### 1.4 Runtime verification

Create a minimal deterministic snippet per API and record shape, dtype, device, and checksum. Save stdout to `work_log/<run_id>/runtime_checks.txt`.

**NumPy**

```python
import numpy as np, hashlib
def checksum(a): import hashlib; return hashlib.sha256(a.tobytes()).hexdigest()[:16]
rng = np.random.default_rng(12345)
x = rng.normal(size=(1000,), dtype=np.float64)
print("shape", x.shape, "dtype", x.dtype, "sum", float(x.sum()))
print("sha256-16", checksum(x))
```

**PyTorch**

```python
import torch, hashlib
def checksum(t): import hashlib; return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]
torch.manual_seed(12345)
torch.use_deterministic_algorithms(True)
x = torch.randn(1000, dtype=torch.float64, device="cpu")
print("shape", tuple(x.shape), "dtype", x.dtype, "device", x.device)
print("sha256-16", checksum(x))
```

**Gradient checks** (for differentiable APIs)

```python
import torch
from torch.autograd import gradcheck
torch.manual_seed(0)
fn = lambda z: (z**2).sum()
inp = (torch.randn(5, dtype=torch.float64, requires_grad=True),)
print("gradcheck", gradcheck(fn, inp, eps=1e-6, atol=1e-8, rtol=1e-6))
```

---

## 2) Plan

### 2.1 Design sketch

Describe intended changes, affected files, new helpers, and invariants. Provide a compatibility plan if public behavior changes and a migration note if needed.

### 2.2 Test matrix and criteria

Define unit, integration, and reproducibility tests. Add performance only after equivalence is proven.

| Test id | Type | Assertion | Tolerance | Devices | Seed | Notes |
|---|---|---|---|---|---|---|
| T1 | unit | helper returns correct shape and dtype | atol=1e-8, rtol=1e-6 | cpu | 12345 | gradcheck if applicable |
| T2 | integration | end-to-end metrics equal baseline | same tolerance | cpu, accelerator | 12345 | baseline fixture |
| T3 | reproducibility | two runs equal for fixed seeds | exact or strict tolerance | cpu, accelerator | 12345 | log versions |
| T4 | perf (post-equivalence) | speed vs baseline | report wall time | cpu, accelerator | fixed | isolate I/O |

Determinism for CUDA when present. Set `torch.use_deterministic_algorithms(True)`. For cuBLAS kernels on CUDA 10.2+, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` or `:16:8` in the environment before Python starts.

### 2.3 Documentation

Update docstrings with shapes, dtypes, device policies, and error modes. Update `CHANGELOG.md` and add a migration note if behavior changes.

---

## 3) Code

Principles. Clear names, repository style, explicit dtype and device at boundaries, and no ad hoc guards. Prefer passing `numpy.random.Generator` instances and avoid global random state. For PyTorch, set seeds and request deterministic kernels.

---

## 4) Test

Run the full suite. If a test fails, return to Plan to fix the root cause.

### Work log policy

All ephemeral files live under a run-scoped directory.

- Base: `work_log/`
- Run id: `YYYYMMDDTHHMMSSZ-shortslug` in UTC
- Path: `work_log/<run_id>/`

Create the directory before any script writes outputs. Store manifests, logs, seeds, and environment captures there.

Commands

```bash
export RUN_ID=$(date -u +"%Y%m%dT%H%M%SZ")-research
mkdir -p work_log/$RUN_ID
python runtime_check_numpy.py  | tee work_log/$RUN_ID/runtime_checks.txt
python runtime_check_torch.py  | tee -a work_log/$RUN_ID/runtime_checks.txt
python - << 'PY' > work_log/$RUN_ID/env.txt
import platform, sys, subprocess
print('python', sys.version); print('platform', platform.platform())
try:
    import torch; print('torch', torch.__version__)
except Exception as e:
    print('torch import error', e)
subprocess.run([sys.executable, '-m', 'pip', 'list'])
PY
```

Manifest writer

```python
# scripts/write_manifest.py
import hashlib, json, os, sys
run_dir = sys.argv[1]
def sha256_hex(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): h.update(c)
    return h.hexdigest()
entries=[]
for root,_,files in os.walk(run_dir):
    for fn in files:
        p=os.path.join(root,fn); st=os.stat(p)
        entries.append({{'path': os.path.relpath(p, run_dir),'bytes': st.st_size,'sha256': sha256_hex(p)}})
with open(os.path.join(run_dir,'MANIFEST.json'),'w') as f:
    json.dump({{'run_dir': run_dir, 'files': entries}}, f, indent=2)
print('wrote', os.path.join(run_dir,'MANIFEST.json'))
```

`.gitignore`

```
work_log/*/
!work_log/**/MANIFEST.json
!work_log/**/README.md
!work_log/**/runtime_checks.txt
!work_log/**/env.txt

__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
```

---

## 5) Write-up

Provide a concise PR-ready report.

Goal. What problem this change solves.  
Approach. Design choices and alternatives.  
Verification. Tests, baselines, seeds, versions, and tolerance. Include gradients where relevant.  
Outcome. Speedups, simplifications, or guarantees with numbers.  
Useful commands. Keep profiling and test commands that matter.  
Iteration log. Short cycle summaries with goal, change set, and results.

---

## Active to-do board

| ID | Task | Status | Owner | Evidence | Next action | Due | Done at |
|---|---|---|---|---|---|---|---|
| E1 | Fill API checklist with real symbols | todo | assignee | links to defs | open files, confirm signatures and error modes | +3d | |
| E2 | Runtime verification snippets per API | todo | assignee | stdout in work_log | save checksums | +3d | |
| E3 | Add tests T1–T3 | todo | assignee | PR tests | implement and run in CI | +4d | |
| E4 | Update docs and changelog | todo | assignee | PR diff | write migration note if needed | before release | |

---

## References

Keep these nearby when resolving standards or disputes.
