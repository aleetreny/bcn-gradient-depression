# The team harness

**Branch `harness`.** This is one shared way to build, judge, compare and record a possession pipeline, so that results from different people (and different coding agents) can be compared, reproduced and trusted.

## One harness on every branch

The harness ships from `main`, and every branch takes it from there by merging `main`. The files are the same everywhere; what runs depends on what the branch has:

| Layer | Needs | `main`, `Bernardo` (no research code) | `Ale` and its descendants |
|---|---|---|---|
| AI-assistant safety: `AGENTS.md` and its pointers, the agent hook, `guard`, `check`, git hooks, CI | git, Python 3.11, numpy | **full** | **full** |
| Experiment harness: specs, judges J0–J3, gate, runner | `src/` research code (+ local data for J1–J3) | listed as *unavailable: needs src/…*; its tests are skipped | full |

So a branch can't opt out of the safety layer, and nothing crashes on a branch that hasn't taken the research code. The file lists in `harness/policy.py` name files that exist on some branches only; a missing file is simply never matched.

## Why we need it

The research so far has been strong, but the way we work has started to limit it:

| Symptom (from this repo) | Cost |
|---|---|
| 79 one-off scripts in `scripts/`, the largest over 1,000 lines, each re-implementing load → degrade → features → fit → decode → score | The same bug can be fixed in one script and not the others. Two "J1" numbers from different scripts are not guaranteed to measure the same thing. |
| Judges defined three times (`src/eval/metrics.lomo`, `lab/tune.judge`, `scripts/s3_7_nested_full.py`) | Hard to tell which protocol produced a given number. |
| v4 → v7 added about 7 models and 4 stages for +0.037 LOMO, with many steps of +0.001 to +0.003 against a fold std of about 0.03 (`research/MODEL_V8_DESIGN.md`) | We stack changes that 7 folds cannot tell apart from noise. |
| Decision criteria written after the results (except F4.10, which pre-registered them) | Easy to fool ourselves without meaning to. |
| `.gitignore` is the only thing stopping gated data from being committed | One `git add -f` ends the competition for us. |

## Architecture

```
 experiments/specs/<id>.toml  ── the hypothesis + pre-registered criteria (committed BEFORE the run)
            │
            ▼
 ┌──────────────────────────── harness/runner.py ────────────────────────────┐
 │  registry ──► predictor = features ─► model ─► decoder   (harness/components.py)
 │                  │contracts│       │contracts│ │contracts│  (harness/contracts.py)
 │  judges   ──► J0 synthetic · J1 Sportec LOMO · J2 Metrica Sample · J3 public BAR-ATH
 │  scoring  ──► src/eval/metrics.score  (official evaluate(), unchanged)
 │  gate     ──► paired per-fold Δ vs champion → ACCEPT / NOISE / REJECT
 └────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
 experiments/runs/<id>/result.json   metrics + spec + commit + source digest   (committed)
 experiments/runs/CHAMPION           the run every new spec is gated against   (committed)
 experiments/registry.csv            optional row (--register), keeps the old log going
 local_data/harness/<id>/            predictions (--save-pred)                  (never committed)
```

The layers depend only downwards. `src/` does not import `harness/`, so the frozen v4 pipeline and `scripts/reproduce_final.sh` are untouched.

### 1. Contracts (`harness/contracts.py`)

Each stage boundary has one check. They are cheap and always on, so a bug shows up at the stage that caused it.

| Stage | Output | Check |
|---|---|---|
| features | DataFrame with T rows, no ±inf, no duplicate columns | `check_features` |
| model | (T, 3) probabilities H, A, D; finite, rows sum to 1 | `check_proba` |
| decoder / predictor | (T,) int8 in {0, 1, 2} | `check_labels` |

There is also a symmetry contract: relabelling H↔A must relabel the output and change nothing else. `tests/test_harness.py` runs every registered decoder through it automatically.

### 2. Registry (`harness/registry.py`, `harness/components.py`)

Components are registered by name, so a spec can say `decoder = "hsmm"` without a script having to import it.

```python
@register("decoder", "soft_hsmm")
def soft_hsmm(proba, half, params, change=None): ...
```

| Kind | Signature |
|---|---|
| `features` | `fn(match) -> DataFrame` |
| `model` | `factory(**params)` → `.fit(Xs, ys, groups)`, `.predict_proba(X)` |
| `decoder` | `fn(proba, half, params, change=None) -> int8` (the same signature as `lab/decoders.py`) |
| `predictor` | `factory(**params) -> fit_predict(train: dict[str, Match], test: Match) -> int8` |

`staged` composes features → model → decoder by name. Any pipeline that doesn't fit that shape, such as an event lattice (v8) or a sequence model on raw tracking, registers as its own `predictor`. The judges only ever see `fit_predict`.

### 3. Judges (`harness/judges.py`)

These are the only places a number that drives a decision may come from.

| Judge | What | Role |
|---|---|---|
| **J0** | LOMO on 4 synthetic matches | plumbing; CI runs it. Never a selection signal. |
| **J1** | LOMO on the 7 Sportec matches, degraded with `configs/degrade_v3.json` (test copy seed 0, training copies `train_seeds`) | **select**: the primary decision |
| **J2** | Train on all of degraded Sportec, test on Metrica Sample 1–3 (same provider as BAR-ATH) | **guard**: may veto |
| **J3** | Train on all of degraded Sportec, score BAR-ATH on the public 5 min | **report** only (17 possessions) |

A judge whose data is missing reports `unavailable` and is skipped; it is never faked. Judge parameters are part of the result, and two runs are only gated against each other when their judge parameters match. Otherwise the result is `INCOMPARABLE`.

### 4. Gate (`harness/gate.py`)

The comparison is paired by fold, so match difficulty cancels out: `d_i = spellF1(candidate, i) − spellF1(champion, i)`.

- **ACCEPT** needs all three:
  - mean d ≥ `min_delta` (default 0.01, about one standard error on J1);
  - at least 70 % of folds improved;
  - the paired-bootstrap 90 % interval above 0.
- **REJECT**: mean d ≤ 0, and the interval rules out a gain of `min_delta`.
- **NOISE**: anything else. It may be real, but these folds can't tell. **Don't stack it**: combine several NOISE ideas into one spec and test them together.

The overall decision needs every *select* judge to ACCEPT and no *guard* judge to REJECT. The criteria live in the spec and are committed before the run, so changing them afterwards shows up in `git log`.

### 5. Records (`harness/runner.py`)

`result.json` stores per-fold and summary metrics, the spec and its digest, the commit, a list of dirty files, a digest of the `src/` and `harness/` sources, and the Python version and platform. A run from a dirty tree is marked as such, and `promote` refuses it.

### 6. Guards (`harness/guards.py`, `.githooks/pre-commit`, CI)

`python -m harness guard` fails if any tracked or staged file:
- matches a gated or private pattern (ATD, Smart Tagging, homography, full GT, `data/`, `submissions/`, `local_data/`, model binaries, `.env`); or
- is larger than 5 MB.

It runs in CI and, once you run `git config core.hooksPath .githooks`, on every commit.

## AI-assistant safety layer

Everyone on the team codes with an AI assistant, so the harness assumes that **the agent writes most of the code and runs most of the commands**. The safety cannot depend on the agent remembering or agreeing to a rule, so it is built in three layers:

| Layer | What | Depends on the agent cooperating? |
|---|---|---|
| **Instructions** | `AGENTS.md`, auto-loaded by every common assistant (`CLAUDE.md`, `GEMINI.md`, `.cursor/rules/harness.mdc`, `.github/copilot-instructions.md` all point to it) | yes: they tell it the protocol and how to report |
| **Definition of done** | `python -m harness check` → OK / FAIL / REVIEW. Checks leaks, protected files changed, an append-only registry, run records backed by a pre-registered spec, lint and tests | partly: the agent is told to run it, and the pre-push hook and CI run it anyway |
| **Hard blocks** | `harness/agent_hook.py` (a Claude Code PreToolUse hook, in the committed `.claude/settings.json`) blocks before execution: `--no-verify`, `git add -f`, force-push, history rewrites, hook tampering, bulk discards, uploads of gated data, recursive deletes of data, writes to label files, run records and protected files. `.githooks/` and CI catch the same things for every other tool. | no |

The policy lives in one file, `harness/policy.py` (PROTECTED, RUNNER_ONLY, NO_HAND_LABELS), shared by the hook and by `check`. **REVIEW** means a protected file changed (scorer, competition documents, frozen v4 inputs, the harness's own safety). It is allowed, but the agent must stop and ask. Only a human passes `--allow-protected`.

Limits, stated honestly:
- The pre-tool hook exists only for Claude Code. Other assistants are caught at commit, push and CI time rather than before the action.
- A command can be obfuscated past regexes. The git hooks and CI are the backstop, because they look at what actually changed, not at the command.
- Git hooks need `git config core.hooksPath .githooks` once per clone.

## The workflow (for people and agents)

```bash
python -m harness new h012_soft_hsmm            # 1. scaffold the spec
$EDITOR experiments/specs/h012_soft_hsmm.toml   # 2. hypothesis, components, criteria
git add experiments/specs/h012_soft_hsmm.toml && git commit -m "spec: h012 soft hsmm"   # 3. pre-register
python -m harness run experiments/specs/h012_soft_hsmm.toml --register                 # 4. judge + gate
git add experiments/runs/h012_soft_hsmm/result.json experiments/registry.csv           # 5. record
python -m harness promote h012_soft_hsmm        # 6. only on ACCEPT, from a clean tree
```

Write the `experiments/log/NNN_*.md` entry as before. The harness supplies the numbers; the log supplies the *why*.

## Testing pyramid

| Tier | Where | Data | Runs |
|---|---|---|---|
| Unit | `tests/test_*.py` | synthetic only | CI, every push |
| Contract | `tests/test_harness.py` (parametrised over the registry) | synthetic | CI; new components are covered automatically |
| Smoke | `experiments/specs/smoke_synthetic.toml` on J0 | synthetic | CI |
| Judge | `python -m harness run <spec>` on J1–J3 | Sportec, Sample, BAR-ATH (local only) | developer machine |
| Regression | `bash scripts/reproduce_final.sh` (SHA-256 of v4) | competition data | before any freeze |

Rule of thumb: if a bug could be caught on synthetic data, write that test. CI can never see real data.

## Migration plan (incremental; nothing breaks)

1. **Now (this branch):** harness + baseline (`h000`) and v1 GBM + HSMM (`h001`) as reference specs. `h000` is the first champion.
2. **Register what already exists:** the v4 pieces (`v1ctl` features, local TCN, transition head, M3 decoder) as components, then the v4 submission as a spec. Its J1 must match registry row 029 within noise. That is the harness's acceptance test.
3. **Fold in `lab/`:** register `lab/decoders.py` (soft HSMM) and move the cross-fitted decoder tuning from `lab/tune.py` into a `tuned` decoder wrapper, so tuning happens inside each fold.
4. **New work goes through specs only** (v8 event lattice, `research/MODEL_V8_DESIGN.md`). Old `scripts/` stay as historical record and are not ported unless they are needed again.
5. **Optional later:** nested J1 (decoder tuned on inner folds), fold-level caching keyed by `src/eval/store.identity`, and a parallel fold runner.

## Team roles

| Role | Owns |
|---|---|
| Harness owner | `harness/`, judges, gate defaults, CI. Changes to judges or gate need a PR approved by a second person. |
| Component authors | One adapter in `harness/components.py` (or their own module) + one spec per idea |
| Reviewer | Checks the PR template: spec committed before the result, gate decision quoted, no gated files |
