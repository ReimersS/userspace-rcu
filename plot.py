#!/usr/bin/env python3
"""
plot.py — Benchmark plots for orb fence-cost synthesis evaluation.

Reads SUMMARY lines from .tap files in a runs/ directory.
Produces figures suitable for an academic paper (double column).

Usage:
    python3 plot.py runs/2026-07-31_12-10-31
    python3 plot.py runs/2026-07-31_12-10-31 --output figs/bench
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paper geometry and style
# ---------------------------------------------------------------------------

SINGLE_W = 3.3  # single-column width (inches)
DOUBLE_W = 7.0  # double-column width (inches)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.2,
    "errorbar.capsize": 2,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ---------------------------------------------------------------------------
# Compiler styling — single source of truth
# ---------------------------------------------------------------------------

PAL = sns.color_palette("pastel", 10)
HATCHES = ["", "//", "\\\\", "xx", "..", "oo"]

# fc value → size label (ascending fc = more relaxed = larger size)
FC_SIZES = {1: "S", 2: "S", 256: "M", 333: "M", 666: "L", 999: "XL"}

# Canonical label → (color, hatch).
LABEL_STYLE = {
    "clang":   (PAL[0], HATCHES[0]),
    "clangir": (PAL[1], HATCHES[1]),
    "orb XL":  (PAL[2], HATCHES[2]),
    "orb L":   (PAL[3], HATCHES[3]),
    "orb M":   (PAL[4], HATCHES[4]),
    "orb S":   (PAL[5], HATCHES[5]),
}


def _label(compiler):
    """Short display label for a compiler string."""
    if compiler.startswith("clangir-"):
        return "clangir"
    if compiler.startswith("clang-"):
        return "clang"
    m = re.search(r"fc(\d+)", compiler)
    if m:
        return f"orb {FC_SIZES.get(int(m.group(1)), 'fc' + m.group(1))}"
    return compiler


def _style(compiler):
    """Return (color, hatch) for a compiler, registering unknown ones."""
    lbl = _label(compiler)
    if lbl not in LABEL_STYLE:
        idx = len(LABEL_STYLE)
        LABEL_STYLE[lbl] = (PAL[min(idx, len(PAL) - 1)],
                            HATCHES[min(idx, len(HATCHES) - 1)])
    return LABEL_STYLE[lbl]


def _color(compiler):
    return _style(compiler)[0]


def _hatch(compiler):
    return _style(compiler)[1]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SUMMARY_RE = re.compile(
    r"SUMMARY\s+\S*test_(\w+)\s+"
    r"testdur\s+(\d+)\s+"
    r"nr_readers\s+(\d+)\s+"
    r"rdur\s+(\d+)\s+"
    r"wdur\s+(\d+)\s+"
    r"nr_writers\s+(\d+)\s+"
    r"wdelay\s+(\d+)\s+"
    r"nr_reads\s+(\d+)\s+"
    r"nr_writes\s+(\d+)\s+"
    r"nr_ops\s+(\d+)"
    r"(?:\s+batch\s+(\d+))?"
)


def load_runs(run_dir: Path) -> pd.DataFrame:
    rows = []
    for compiler_dir in sorted(run_dir.iterdir()):
        if not compiler_dir.is_dir():
            continue
        compiler = compiler_dir.name
        for tap in sorted(compiler_dir.glob("*.tap")):
            parts = tap.stem.split(".")
            run = int(parts[-1]) if len(parts) >= 3 else 1
            for line in tap.read_text().splitlines():
                m = SUMMARY_RE.search(line)
                if not m:
                    continue
                test, dur = m.group(1), int(m.group(2))
                nr, rdur, wdur = int(m.group(3)), int(m.group(4)), int(m.group(5))
                nw, wdelay = int(m.group(6)), int(m.group(7))
                reads, writes, ops = int(m.group(8)), int(m.group(9)), int(m.group(10))
                batch = int(m.group(11)) if m.group(11) else 0
                rows.append({
                    "compiler": compiler, "test": test, "run": run,
                    "duration": dur, "nr_readers": nr, "nr_writers": nw,
                    "rdur": rdur, "wdur": wdur, "wdelay": wdelay,
                    "batch": batch,
                    "reads": reads, "writes": writes, "ops": ops,
                    "reads_per_sec": reads / dur, "writes_per_sec": writes / dur,
                    "ops_per_sec": ops / dur,
                })
    df = pd.DataFrame(rows)
    dedup_cols = ["compiler", "test", "run", "batch", "nr_readers",
                  "nr_writers", "wdelay", "rdur", "wdur"]
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    df["opt"] = df["compiler"].str.extract(r"-(O\d)")
    df["fc"] = df["compiler"].str.extract(r"-fc(\d+)").astype(float)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compilers_for_opt(df, opt):
    """Return ordered list of compilers present in df for a given opt level.
    Order: clang, clangir, then orb sorted by fc descending."""
    present = set(df[df["opt"] == opt]["compiler"].unique())
    order = []
    for prefix in [f"clang-{opt}", f"clangir-{opt}"]:
        if prefix in present:
            order.append(prefix)
    orb = sorted([c for c in present if c.startswith(f"orb-{opt}")],
                 key=lambda c: int(m.group(1))
                 if (m := re.search(r"fc(\d+)", c)) else 0)
    order.extend(orb)
    return order


def _dedup_legend(axes):
    """Collect legend handles from multiple axes, deduplicated by label."""
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen[l] = h
    return list(seen.values()), list(seen.keys())


def _save(fig, base, suffix):
    if base:
        for ext in (".pdf", ".png"):
            fig.savefig(f"{base}_{suffix}{ext}")


def _billions(ax):
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e9:.1f}"))


def _millions(ax):
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))


TEST_LABELS = {
    "mutex": "mutex", "rwlock": "rwlock", "perthreadlock": "ptlock",
    "urcu": "urcu", "urcu_mb": "mb", "urcu_qsbr": "qsbr",
    "urcu_gc": "gc", "urcu_mb_gc": "mb_gc", "urcu_qsbr_gc": "qsbr_gc",
    "urcu_lgc": "lgc", "urcu_mb_lgc": "mb_lgc", "urcu_qsbr_lgc": "qsbr_lgc",
}


# ---------------------------------------------------------------------------
# Plot 1: Read & write throughput vs batch size — line plot, O3/O0 columns
# ---------------------------------------------------------------------------

def plot1_throughput_vs_batch(df, base):
    filt = ((df["test"] == "urcu_gc") & (df["batch"] > 0)
            & (df["nr_readers"] == 32) & (df["nr_writers"] == 32)
            & (df["wdelay"] == 0))
    d_o3 = df[filt & (df["opt"] == "O3")]
    d_o0 = df[filt & (df["opt"] == "O0")]
    if d_o3.empty and d_o0.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(SINGLE_W, 3.0),
                              sharex=True, sharey="row")
    ax_r0, ax_r3 = axes[0]
    ax_w0, ax_w3 = axes[1]

    for ax_r, ax_w, opt_df, opt in [(ax_r3, ax_w3, d_o3, "O3"),
                                     (ax_r0, ax_w0, d_o0, "O0")]:
        compilers = _compilers_for_opt(df, opt)
        for c in compilers:
            cd = opt_df[opt_df["compiler"] == c]
            if cd.empty:
                continue
            col = _color(c)
            for ax, metric in [(ax_r, "reads_per_sec"),
                               (ax_w, "writes_per_sec")]:
                agg = cd.groupby("batch")[metric].agg(["mean", "std"]).reset_index()
                ax.plot(agg["batch"], agg["mean"], label=_label(c), color=col)
                ax.fill_between(agg["batch"],
                                agg["mean"] - agg["std"],
                                agg["mean"] + agg["std"],
                                color=col, alpha=0.2)

    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        sns.despine(ax=ax)

    ax_r0.set_ylabel("Reads / s (×10⁹)", fontsize=7)
    ax_w0.set_ylabel("Writes / s (×10⁶)", fontsize=7)
    _billions(ax_r3); _billions(ax_r0)
    _millions(ax_w3); _millions(ax_w0)

    ax_r0.set_title("O0", fontsize=8, fontweight="bold")
    ax_r3.set_title("O3", fontsize=8, fontweight="bold")
    ax_w3.set_xlabel("Batch size")
    ax_w0.set_xlabel("Batch size")

    fig.suptitle("Throughput vs. batch size, urcu_gc (32r/32w, wdelay=0)",
                 fontsize=9, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.6)
    fig.align_ylabels(axes[:, 0])

    handles, labels = _dedup_legend([ax_r3, ax_r0])
    if handles:
        fig.legend(handles, labels, fontsize=6, ncol=len(handles),
                   loc="upper center", bbox_to_anchor=(0.5, 0.98),
                   frameon=False, handlelength=1.5, columnspacing=1.0)
    _save(fig, base, "1_throughput_batch")


# ---------------------------------------------------------------------------
# Plot 2: Normalized throughput across benchmarks — grouped bars, O3/O0 rows
# ---------------------------------------------------------------------------

PLOT2_TESTS = [
    ("mutex",        0),
    ("rwlock",       0),
    ("urcu",         0),
    ("urcu_mb",      0),
    ("urcu_qsbr",    0),
    ("urcu_gc",      1),
    ("urcu_mb_gc",   32768),
    ("urcu_qsbr_gc", 32768),
]


def _plot_normalized_multi(ax, df_opt, tests, compilers):
    """Grouped bars: each compiler normalized to clang(ir) baseline."""
    if not compilers:
        return

    # Determine baseline: prefer clangir > clang > lowest-fc orb
    opt = compilers[0].split("-")[1]
    baseline = None
    for candidate in [f"clangir-{opt}", f"clang-{opt}"]:
        if candidate in df_opt["compiler"].unique():
            baseline = candidate
            break
    if baseline is None:
        orb = [c for c in compilers if "fc" in c]
        baseline = min(orb, key=lambda c: int(c.split("fc")[-1])) if orb else compilers[0]

    n_tests, n_comp = len(tests), len(compilers)
    w = 0.8 / n_comp
    group_x = np.arange(n_tests)

    for j, c in enumerate(compilers):
        vals = []
        for test, batch in tests:
            td = df_opt[(df_opt["test"] == test) & (df_opt["batch"] == batch)]
            c_mean = td[td["compiler"] == c]["reads_per_sec"].mean()
            b_mean = td[td["compiler"] == baseline]["reads_per_sec"].mean()
            vals.append(c_mean / b_mean if pd.notna(c_mean) and b_mean > 0 else np.nan)
        offset = (j - (n_comp - 1) / 2) * w
        ax.bar(group_x + offset, vals, w * 0.9,
               color=_color(c), edgecolor="black", linewidth=0.4,
               hatch=_hatch(c), label=_label(c))

    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xticks(group_x)
    ax.set_xticklabels([TEST_LABELS.get(t, t) for t, _ in tests],
                       rotation=30, ha="right")
    ax.set_ylim(bottom=0)


def plot2_normalized_throughput(df, base):
    d = df[df["nr_readers"] == 32]
    tests = [(t, b) for t, b in PLOT2_TESTS
             if not d[(d["test"] == t) & (d["batch"] == b)].empty]
    if not tests:
        return

    fig, (ax3, ax0) = plt.subplots(2, 1, figsize=(DOUBLE_W, 4.0), sharey=True)

    for ax, opt in [(ax3, "O0"), (ax0, "O3")]:
        compilers = _compilers_for_opt(df, opt)
        _plot_normalized_multi(ax, d[d["opt"] == opt], tests, compilers)
        ax.set_title(opt, fontsize=8, fontweight="bold")
        ax.set_ylabel("Norm. reads")
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        sns.despine(ax=ax)

    fig.suptitle("Normalized read throughput (32r/32w)",
                 fontsize=9, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.8)

    handles, labels = _dedup_legend([ax3, ax0])
    if handles:
        fig.legend(handles, labels, fontsize=6, ncol=len(handles),
                   loc="upper center", bbox_to_anchor=(0.5, 0.97),
                   frameon=False, handlelength=1.5, columnspacing=1.0)
    _save(fig, base, "2_normalized")


# ---------------------------------------------------------------------------
# Plot 3: Reader scalability — reads/s vs nr_readers (0 writers, rdur=0)
# ---------------------------------------------------------------------------

PLOT3_TESTS = ["mutex", "rwlock", "urcu", "urcu_mb", "urcu_qsbr",
               "urcu_gc", "urcu_mb_gc", "urcu_qsbr_gc"]


def plot3_reader_scalability(df, base):
    d = df[(df["nr_writers"] == 0) & (df["rdur"] == 0)]
    if d.empty:
        return

    tests = [t for t in PLOT3_TESTS if t in d["test"].unique()]
    if not tests:
        return

    n_tests = len(tests)
    fig = plt.figure(figsize=(DOUBLE_W, 3.6))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.65,
                              top=0.78, bottom=0.10, left=0.07, right=0.98)

    all_axes = []
    for opt_idx, opt in enumerate(["O0", "O3"]):
        inner = gridspec.GridSpecFromSubplotSpec(1, n_tests,
                                                 subplot_spec=outer[opt_idx])
        d_opt = d[d["opt"] == opt]
        compilers = _compilers_for_opt(d, opt)

        row_axes = []
        for t_idx, test in enumerate(tests):
            ax = fig.add_subplot(inner[t_idx],
                                 sharey=row_axes[0] if row_axes else None)
            row_axes.append(ax)
            td = d_opt[d_opt["test"] == test]

            for c in compilers:
                cd = td[td["compiler"] == c]
                if cd.empty:
                    continue
                agg = cd.groupby("nr_readers")["reads_per_sec"].agg(
                    ["mean", "std"]).reset_index()
                ax.plot(agg["nr_readers"], agg["mean"],
                        label=_label(c), color=_color(c))
                ax.fill_between(agg["nr_readers"],
                                agg["mean"] - agg["std"],
                                agg["mean"] + agg["std"],
                                color=_color(c), alpha=0.15)

            ax.grid(True, alpha=0.3, linewidth=0.5)
            sns.despine(ax=ax)
            ax.set_title("", pad=0)
            ax.set_xlabel("")
            ax.tick_params(labelsize=6)
            if t_idx > 0:
                plt.setp(ax.get_yticklabels(), visible=False)

        row_axes[0].set_ylabel("Reads / s")
        for ax in row_axes:
            _billions(ax)
        all_axes.extend(row_axes)

        pos = outer[opt_idx].get_position(fig)
        if opt_idx == 0:
            for t_idx, test in enumerate(tests):
                ax_pos = row_axes[t_idx].get_position()
                cx = (ax_pos.x0 + ax_pos.x1) / 2
                fig.text(cx, pos.y1 + 0.01, TEST_LABELS.get(test, test),
                         ha="center", va="bottom", fontsize=7)
        fig.text(0.5, pos.y1 + 0.05, opt, ha="center", va="bottom",
                 fontsize=8, fontweight="bold")

    fig.suptitle("Reader scalability (0 writers, rdur=0)",
                 fontsize=9, fontweight="bold", y=0.99)
    fig.supxlabel("Readers", fontsize=8)

    handles, labels = _dedup_legend(all_axes)
    if handles:
        fig.legend(handles, labels, fontsize=6, ncol=len(handles),
                   loc="upper center", bbox_to_anchor=(0.5, 0.96),
                   frameon=False, handlelength=1.5, columnspacing=1.0)
    _save(fig, base, "3_reader_scalability")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark plots for orb evaluation")
    ap.add_argument("rundir", help="Path to runs directory")
    ap.add_argument("--output", "-o", default=None,
                    help="Output base path; per-plot suffixes added")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    df = load_runs(Path(args.rundir))
    print(f"Loaded {len(df)} data points from {args.rundir}")
    print(f"Compilers: {sorted(df['compiler'].unique())}")
    print(f"Tests: {sorted(df['test'].unique())}")

    plot1_throughput_vs_batch(df, args.output)
    plot2_normalized_throughput(df, args.output)
    plot3_reader_scalability(df, args.output)

    if args.show:
        plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
