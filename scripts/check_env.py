"""Check the research environment: print versions and smoke-test each library.

Run with the research interpreter:
    .venv311/bin/python scripts/check_env.py

Every check is a tiny real operation (fit, solve, write/read), not just an import, so a broken
native dependency (OpenMP for LightGBM, the Arrow C++ runtime) fails here and not mid-experiment.
"""
from __future__ import annotations

import importlib
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "lightgbm", "optuna", "kloppy",
            "matplotlib", "tqdm", "yaml", "pytest")


def smoke_tests() -> dict:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    results = {}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.parquet"
        pd.DataFrame(X, columns=list("abcde")).to_parquet(path)
        results["pyarrow parquet round-trip"] = pd.read_parquet(path).shape == (2000, 5)

    from scipy.optimize import minimize_scalar
    results["scipy optimize"] = abs(minimize_scalar(lambda v: (v - 3) ** 2).x - 3) < 1e-6

    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    p = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
    results["sklearn logistic + isotonic"] = IsotonicRegression(out_of_bounds="clip").fit(p, y).predict(p).shape == (2000,)

    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=20, verbose=-1, n_jobs=2).fit(X, y)
    results["lightgbm fit (OpenMP)"] = (model.predict(X) == y).mean() > 0.9

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(lambda t: (t.suggest_float("x", -5, 5) - 1) ** 2, n_trials=20)
    results["optuna TPE"] = abs(study.best_params["x"] - 1) < 1.0

    from kloppy import metrica, sportec  # noqa: F401  (loaders we use in S0.3/S0.4)
    results["kloppy sportec/metrica loaders"] = True

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with tempfile.TemporaryDirectory() as tmp:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(Path(tmp) / "t.png")
        plt.close(fig)
        results["matplotlib Agg savefig"] = (Path(tmp) / "t.png").stat().st_size > 0
    return results


def main() -> int:
    print("Python   {} ({})".format(platform.python_version(), sys.executable))
    print("Platform {} {} | {}".format(platform.system(), platform.machine(), platform.platform()))
    if sys.version_info[:2] != (3, 11):
        print("WARNING: research environment is meant to be Python 3.11")
    print()
    for name in PACKAGES:
        module = importlib.import_module(name)
        print("  {:<12} {}".format(name, getattr(module, "__version__", "?")))
    try:
        import torch
        print("  {:<12} {} (mps available: {})".format("torch", torch.__version__, torch.backends.mps.is_available()))
    except ImportError:
        print("  {:<12} not installed (added when the first network step needs it)".format("torch"))
    print()
    results = smoke_tests()
    for name, ok in results.items():
        print("  [{}] {}".format("ok" if ok else "FAIL", name))
    failed = [name for name, ok in results.items() if not ok]
    print("\n{}".format("Environment OK." if not failed else "FAILED: " + ", ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
