"""Small shared helpers: plotting style, saving figures/tables, JSON output."""
from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
    "axes.prop_cycle": matplotlib.cycler(color=C.PALETTE), "legend.frameon": False,
})


def savefig(fig, name: str):
    """Save a report figure. Figures are drawn at about the size they are printed (7.2 in for a full-width
    figure printed 5.9 in wide), so the fonts set above stay readable on paper."""
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(C.FIGURES / f"{name}.png")
    plt.close(fig)


def bdt_m(x):
    """Format a BDT amount in millions."""
    return f"{x / 1e6:,.1f}M"


def save_json(obj, name: str):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, pd.Timestamp):
            return str(o.date())
        raise TypeError(type(o))
    with open(C.METRICS / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def load_json(name: str):
    with open(C.METRICS / f"{name}.json") as f:
        return json.load(f)


def latex_escape(s: str) -> str:
    """Escape LaTeX special characters; text that already contains math ($...$) is left untouched."""
    if "$" in str(s):
        return str(s)
    rep = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(rep.get(ch, ch) for ch in str(s))


def to_latex_table(df: pd.DataFrame, name: str, caption: str, label: str, fmt: dict | None = None,
                   col_align: str | None = None, note: str | None = None, font: str = r"\small"):
    """Write a booktabs table that the LaTeX report \\input{}s. Results never get typed by hand."""
    fmt = fmt or {}
    cols = list(df.columns)
    align = col_align or ("l" + "r" * (len(cols) - 1))
    pct = lambda s: re.sub(r"(?<!\\)%", r"\\%", s)          # a bare % would comment out the rest of the line
    # adjustbox shrinks a table only if it is wider than the text block
    lines = [r"\begin{table}[htbp]", r"\centering", font, rf"\caption{{{pct(caption)}}}", rf"\label{{{label}}}",
             r"\begin{adjustbox}{max width=\linewidth}", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(rf"\textbf{{{latex_escape(c)}}}" for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if not isinstance(v, str) and pd.isna(v):
                cells.append("--")
            elif c in fmt and not isinstance(v, str):
                cells.append(pct(fmt[c].format(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:,.2f}")
            else:
                cells.append(latex_escape(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        lines.append(rf"\par\smallskip\parbox{{0.92\linewidth}}{{\footnotesize\emph{{Note:}} {pct(note)}}}")
    lines.append(r"\end{table}")
    (C.TEX_TABLES / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    df.to_csv(C.TABLES / f"{name}.csv", index=False)
