"""Timeline plot: ground truth vs prediction (and optional extra lanes) over a frame range."""
from __future__ import annotations

import numpy as np

from src.data.common import AWAY, DEAD, FPS, HOME

COLORS = {"H": "#2a78d6", "A": "#eb6834", "D": "#8f8e88", "ink": "#52514e", "grid": "#e4e3df"}


def timeline(path, lanes, start, stop, title="", extra=None):
    """lanes: [(name, labels)] drawn top to bottom; extra: [(name, bool array)] grey bars below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(start, stop) / FPS
    extra = extra or []
    n = len(lanes) + len(extra)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": COLORS["grid"], "axes.labelcolor": COLORS["ink"],
                         "xtick.color": COLORS["ink"], "ytick.color": COLORS["ink"]})
    fig, ax = plt.subplots(figsize=(12, 0.7 * n + 1.4), constrained_layout=True)
    ticks, names = [], []
    for i, (name, labels) in enumerate(lanes):
        y = n - i
        seg = labels[start:stop]
        for code, key in ((HOME, "H"), (AWAY, "A"), (DEAD, "D")):
            ax.fill_between(t, y - 0.35, y + 0.35, where=seg == code, step="post", color=COLORS[key], lw=0,
                            label=key if i == 0 else None)
        ticks.append(y)
        names.append(name)
    for j, (name, mask) in enumerate(extra):
        y = n - len(lanes) - j
        ax.fill_between(t, y - 0.25, y + 0.25, where=mask[start:stop], step="post", color=COLORS["ink"],
                        alpha=0.45, lw=0)
        ticks.append(y)
        names.append(name)
    ax.set_yticks(ticks, names)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0.4, n + 0.6)
    ax.set_xlabel("time (s)")
    ax.set_title(title, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.25))
    fig.savefig(path, dpi=130)
    plt.close(fig)
