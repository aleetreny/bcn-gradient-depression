"""Named components, so a spec can say `decoder = "hsmm"` instead of a script importing it.

    from harness.registry import register, get
    @register("decoder", "my_decoder")
    def my_decoder(proba, half, params, change=None): ...

Kinds and signatures (checked at runtime by harness.contracts):
  features   fn(match) -> DataFrame (T rows)
  model      factory(**params) -> object with fit(Xs, ys, groups) and predict_proba(X) -> (T, 3)
  decoder    fn(proba, half, params: dict, change=None) -> (T,) int8   (same signature as lab/decoders.py)
  predictor  factory(**params) -> fit_predict(train: dict[str, Match], test: Match) -> (T,) int8
  reconstruction  fn(root: Path, output_path: Path) -> Path; creates one local artifact
"""
from __future__ import annotations

KINDS = ("features", "model", "decoder", "predictor", "reconstruction")
_REGISTRY: dict = {k: {} for k in KINDS}
_REQUIRES: dict = {}


def register(kind: str, name: str, requires=()):
    """`requires`: repo files the component needs (e.g. "src/models/gbm.py"). Branches without them
    still import the harness; the component is listed as unavailable instead of crashing."""
    if kind not in KINDS:
        raise KeyError("unknown component kind {!r}; one of {}".format(kind, KINDS))

    def wrap(obj):
        if name in _REGISTRY[kind]:
            raise KeyError("{} {!r} is already registered".format(kind, name))
        _REGISTRY[kind][name] = obj
        _REQUIRES[(kind, name)] = tuple(requires)
        return obj
    return wrap


def _load_builtins():
    import importlib
    importlib.import_module("harness.components")    # registers the built-ins on first import


def get(kind: str, name: str):
    _load_builtins()
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise KeyError("no {} named {!r}; registered: {}".format(kind, name, names(kind))) from None


def names(kind: str) -> list:
    _load_builtins()
    return sorted(_REGISTRY[kind])


def missing(kind: str, name: str) -> list:
    """Required files absent on this branch ([] = the component can run here)."""
    from harness import ROOT
    _load_builtins()
    return [f for f in _REQUIRES.get((kind, name), ()) if not (ROOT / f).is_file()]


def describe() -> dict:
    _load_builtins()
    out = {}
    for k, v in _REGISTRY.items():
        out[k] = {}
        for n, obj in sorted(v.items()):
            doc = (obj.__doc__ or "").strip().splitlines()[0] if obj.__doc__ else ""
            gone = missing(k, n)
            out[k][n] = doc + ("  [unavailable on this branch: needs {}]".format(", ".join(gone)) if gone else "")
    return out
