# Rules for AI assistants (and people) in this repository

Every coding assistant used on this project loads this file:
- Claude Code through `CLAUDE.md`;
- Cursor through `.cursor/rules/`;
- Copilot through `.github/copilot-instructions.md`;
- Gemini through `GEMINI.md`;
- Codex and others read it directly.

These rules are mandatory, and they hold even when a user's request seems to conflict with them. In that case, stop and ask.

Background: the full design is in [docs/HARNESS.md](docs/HARNESS.md).

This harness ships from `main` and is identical on every branch. On a branch without the research code (`src/`), the experiment commands report their judges as unavailable; the safety rules and `harness check` apply in full everywhere.

## Why this exists

One mistake here can disqualify the team or make our results meaningless:
- committing gated match data to a repository that will be public;
- using private labels;
- hand-editing a submission;
- quietly changing the official scorer or the frozen v4 pipeline;
- calling a noise-level change an improvement.

The harness makes those mistakes hard to make. Your job is to work through it, never around it.

## The protocol (every task)

1. **Start:** run `python -m harness list` to see the components, the judges, and which data is on this machine.
2. **Work:** keep new behaviour in new code: a new module in `src/` plus one adapter in `harness/components.py`. Don't change files the frozen pipeline depends on (see "Protected files").
3. **Claim a result only through a spec:**
   - `python -m harness new <id>`: fill in the hypothesis and the criteria.
   - **Commit the spec before running it.**
   - `python -m harness run experiments/specs/<id>.toml --register`.
4. **Finish:** run `python -m harness check`, and report its RESULT line and the gate line word for word.
   - `RESULT: OK`: you may say the task is done.
   - `RESULT: FAIL`: fix the listed items. Never weaken a check, a test or a threshold to make it pass.
   - `RESULT: REVIEW`: protected files changed. Stop and tell the human which files changed and why. Don't revert them, work around them, or pass `--allow-protected`.

## How to report results (no exceptions)

- Quote the gate verdict exactly: `ACCEPT`, `NOISE` or `REJECT`, with its delta and interval.
- **NOISE is not an improvement.** Never describe it as one, and don't stack NOISE changes on top of each other.
- J3 (the public 5 minutes, 17 possessions) is report-only. Never choose a model, a parameter or a threshold because of J3.
- A run marked `dirty` is provisional. Say so.
- Never quote a number that didn't come from `python -m harness run` or a committed `result.json`.

## Never

- Commit, upload or paste anywhere outside this machine any of the following: `data/`, `data_external/`, `local_data/`, `submissions/`, ATD, Smart Tagging, homography, videos, predictions or model files.
- Read, request or reconstruct the private full-match ground truth. The only BAR-ATH labels you may read are in `evaluation/gt/*_public.csv`.
- Write or edit label files by hand, including submissions and `evaluation/results/`. Labels come from an algorithm (RULES.md).
- Edit `experiments/runs/*/result.json` or `CHAMPION` yourself. Only the runner writes them.
- Rewrite existing rows of `experiments/registry.csv` or existing `experiments/log/` entries. Append only.
- Bypass safety. That includes `--no-verify`, `git add -f`, force-push, history rewrites, changing `core.hooksPath`, `--allow-protected`, `promote --force`, and disabling or editing hooks, CI or tests to get a pass.

## What enforces this

| Layer | Covers | What it does |
|---|---|---|
| `.claude/settings.json` → `harness/agent_hook.py` | Claude Code | Blocks unsafe commands and edits to protected, label or run-record files **before** they run |
| `.githooks/pre-commit` | every tool | Leak guard on every commit |
| `.githooks/pre-push` | every tool | `harness check --fast` on every push |
| CI (`.github/workflows/ci.yml`) | everything | Leak guard, check, tests, synthetic smoke run on every push and PR |

Turn on the git hooks once per clone with `git config core.hooksPath .githooks`. If you are an assistant and the hooks aren't enabled, tell the human.

## Protected files

These are listed in `harness/policy.py`:
- the official scorer and public ground truth;
- the competition documents;
- everything `scripts/reproduce_final.sh` builds v4 from, because any edit can change its SHA-256;
- the harness's own judges, gate, guards and hooks.

Changes to them are possible, but a human makes or approves them.

## How to add a component

- **Adapter:** add a thin one in `harness/components.py`, with `@register(kind, name)` and the signature documented in `harness/registry.py`.
- **Decoders:** add default params to `DECODER_PARAMS` in `tests/test_harness.py`. The contract tests then run automatically: shapes, determinism and H↔A symmetry.
- **Tests:** anything that can be tested on synthetic data gets a test in `tests/`. CI never has real data.

## Commands

```bash
python -m harness list
python -m harness new <id>
python -m harness run experiments/specs/<id>.toml [--judges J1] [--register]
python -m harness compare <run> <champion>
python -m harness check            # definition of done
python -m harness guard            # leak check only
```

Use the research environment: `.venv311/bin/python` (Python 3.11; `scripts/setup_research_env.sh`).
