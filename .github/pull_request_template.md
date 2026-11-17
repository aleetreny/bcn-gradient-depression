## What and why

<!-- One paragraph. Link the spec and the experiments/log entry. -->

## Evidence (skip for refactors, docs or tooling)

- Spec: `experiments/specs/<id>.toml`, committed before the run (commit: ...)
- Gate vs champion `<champion id>`:
  ```
  J1 vs ...: ACCEPT/NOISE/REJECT delta ...  90% CI [...]  folds improved .../7
  J2 vs ...:
  decision: ...
  ```
- J3 (public 5 min, report only): spellF1 ...

## Checklist

- [ ] `python -m harness guard` passes (no gated, private or heavy files)
- [ ] `python -m pytest -q tests` and `python -m pyflakes src tests harness` pass
- [ ] New components have contract or unit tests on synthetic data
- [ ] Frozen v4 behaviour in `src/` unchanged, or `scripts/reproduce_final.sh` re-run (hash: ...)
- [ ] `experiments/runs/<id>/result.json` and the log entry are committed
