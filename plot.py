#!/usr/bin/env python3
"""
plot.py — Benchmark plots for orb fence-cost synthesis evaluation.

Usage:
    python3 plot.py runs/2026-07-31_12-10-31
    python3 plot.py runs/2026-07-31_12-10-31 --output figs/bench
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paper geometry
# ---------------------------------------------------------------------------

SINGLE_W = 3.3
DOUBLE_W = 7.0

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
# Compiler styling
# ---------------------------------------------------------------------------

PAL = sns.color_palette("pastel", 10)
HATCHES = ["", "//", "\\\\", "xx", "..", "oo"]

FC_SIZES = {1: "S", 333: "M", 666: "L"}

LABEL_STYLE = {
    "clangir": (PAL[1], HATCHES[1]),
    "clang":   (PAL[0], HATCHES[0]),
    "orb S":   (PAL[3], HATCHES[3]),
    "orb M":   (PAL[2], HATCHES[2]),
    "orb L":   (PAL[4], HATCHES[4]),
}

TEST_LABELS = {
        "mutex": "mutex", "rwlock": "rwlock", "perthreadlock": "ptlock",
    "urcu": "urcu", "urcu_mb": "mb", "urcu_qsbr": "qsbr",
    "urcu_gc": "gc", "urcu_mb_gc": "mb_gc", "urcu_qsbr_gc": "qsbr_gc",
    "urcu_lgc": "lgc", "urcu_mb_lgc": "mb_lgc", "urcu_qsbr_lgc": "qsbr_lgc",
    "urcu_hash": "hashtable",
}

LEGEND_KW = dict(fontsize=6, frameon=False, handlelength=1.5, columnspacing=1.0)


def _label(compiler):
    if compiler.startswith("clangir-"):
        return "clangir"
    if compiler.startswith("clang-"):
        return "clang"
    m = re.search(r"fc(\d+)", compiler)
    if m:
        return f"orb {FC_SIZES.get(int(m.group(1)), 'fc' + m.group(1))}"
    return compiler


def _style(compiler):
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
    r"(?:wdur\s+(\d+)\s+)?"
    r"nr_writers\s+(\d+)\s+"
    r"wdelay\s+(\d+)\s+"
    r"nr_reads\s+(\d+)\s+"
    r"nr_writes\s+(\d+)\s+"
    r"nr_ops\s+(\d+)"
    r"(?:\s+batch\s+(\d+))?"
)


_TAP_RE = re.compile(r"\.(\d+)t(?:\.(\d+)c)?(?:\.\d+)?\.tap$")


def load_runs(run_dir: Path) -> pd.DataFrame:
    rows = []
    for compiler_dir in sorted(run_dir.iterdir()):
        if not compiler_dir.is_dir():
            continue
        compiler = compiler_dir.name
        for tap in sorted(compiler_dir.glob("*.tap")):
            m_tap = _TAP_RE.search(tap.name)
            n_threads = int(m_tap.group(1)) if m_tap else 64
            n_cpus = int(m_tap.group(2)) if m_tap and m_tap.group(2) else n_threads
            run = 0
            for line in tap.read_text().splitlines():
                m = SUMMARY_RE.search(line)
                if not m:
                    continue
                run += 1
                dur = int(m.group(2))
                reads, writes, ops = int(m.group(8)), int(m.group(9)), int(m.group(10))
                rows.append({
                    "compiler": compiler, "test": m.group(1), "run": run,
                    "n_cpus": n_cpus,
                    "duration": dur,
                    "nr_readers": int(m.group(3)), "nr_writers": int(m.group(6)),
                    "rdur": int(m.group(4)),
                    "wdur": int(m.group(5)) if m.group(5) else 0,
                    "wdelay": int(m.group(7)),
                    "batch": int(m.group(11)) if m.group(11) else 0,
                    "reads": reads, "writes": writes, "ops": ops,
                    "reads_per_sec": reads / dur,
                    "writes_per_sec": writes / dur,
                    "ops_per_sec": ops / dur,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "compiler", "test", "run", "n_cpus", "duration",
            "nr_readers", "nr_writers",
            "rdur", "wdur", "wdelay", "batch", "reads", "writes", "ops",
            "reads_per_sec", "writes_per_sec", "ops_per_sec", "opt", "fc"])
        return df
    df = df.drop_duplicates(
        subset=["compiler", "test", "run", "n_cpus", "batch", "nr_readers",
                "nr_writers", "wdelay", "rdur", "wdur"],
        keep="first")
    df["opt"] = df["compiler"].str.extract(r"-(O\d)")
    df["fc"] = df["compiler"].str.extract(r"-fc(\d+)").astype(float)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compilers_for_opt(df, opt, include_clang=True):
    present = set(df[df["opt"] == opt]["compiler"].unique())
    order = []
    # clangir first (baseline), then clang, then orb variants
    for prefix in [f"clangir-{opt}"] + ([f"clang-{opt}"] if include_clang else []):
        if prefix in present:
            order.append(prefix)
    orb = sorted([c for c in present if c.startswith(f"orb-{opt}")
                  and (m := re.search(r"fc(\d+)", c))
                  and int(m.group(1)) in FC_SIZES],
                 key=lambda c: int(m.group(1))
                 if (m := re.search(r"fc(\d+)", c)) else 0)
    order.extend(orb)
    return order


def _dedup_legend(axes):
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
        mticker.FuncFormatter(lambda x, _: f"{x / 1e9:.1f}"))


def _millions(ax):
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}"))


def _add_legend(fig, axes, y=0.97):
    handles, labels = _dedup_legend(axes)
    if handles:
        fig.legend(handles, labels, ncol=len(handles),
                   loc="upper center", bbox_to_anchor=(0.5, y), **LEGEND_KW)


# ---------------------------------------------------------------------------
# Plot 1: Throughput vs. batch size (line plot, O0/O3 columns)
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
        for c in _compilers_for_opt(df, opt):
            cd = opt_df[opt_df["compiler"] == c]
            if cd.empty:
                continue
            col = _color(c)
            for ax, metric in [(ax_r, "reads_per_sec"),
                               (ax_w, "writes_per_sec")]:
                agg = cd.groupby("batch")[metric].agg(
                    median="median",
                    q25=lambda x: x.quantile(0.25),
                    q75=lambda x: x.quantile(0.75)).reset_index()
                ax.plot(agg["batch"], agg["median"], label=_label(c), color=col)
                ax.fill_between(agg["batch"], agg["q25"], agg["q75"],
                                color=col, alpha=0.2)

    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        sns.despine(ax=ax)

    ax_r0.set_ylabel("Reads / s (\u00d710\u2079)", fontsize=7)
    ax_w0.set_ylabel("Writes / s (\u00d710\u2076)", fontsize=7)
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
    _add_legend(fig, [ax_r3, ax_r0], y=0.98)
    _save(fig, base, "1_throughput_batch")


# ---------------------------------------------------------------------------
# Plot 2: Normalized read throughput (grouped bars, O0/O3 rows)
# ---------------------------------------------------------------------------

#PLOT2_TESTS = [
#    ("mutex", 0), ("rwlock", 0), ("urcu", 0), ("urcu_mb", 0),
#    ("urcu_qsbr", 0), ("urcu_gc", 1), ("urcu_mb_gc", 32768),
#    ("urcu_qsbr_gc", 32768),
#]

def _plot_normalized_bars(ax, df_sub, tests, compilers, metric="reads_per_sec"):
    """Plot normalized grouped bars for a single opt level / n_cpus slice."""
    if not compilers:
        return
    opt = compilers[0].split("-")[1]
    baseline = None
    for cand in [f"clangir-{opt}", f"clang-{opt}"]:
        if cand in df_sub["compiler"].unique():
            baseline = cand
            break
    if baseline is None:
        orb = [c for c in compilers if "fc" in c]
        baseline = min(orb, key=lambda c: int(c.split("fc")[-1])) if orb else compilers[0]

    n_comp = len(compilers)
    w = 0.8 / n_comp
    group_x = np.arange(len(tests))

    for j, c in enumerate(compilers):
        vals = []
        lo_err = []
        hi_err = []
        for test, batch in tests:
            td = df_sub[(df_sub["test"] == test) & (df_sub["batch"] == batch)]
            c_vals = td[td["compiler"] == c][metric].values
            b_vals = td[td["compiler"] == baseline][metric].values
            b_gmean = np.exp(np.mean(np.log(b_vals[b_vals > 0]))) if np.any(b_vals > 0) else 0
            if len(c_vals) == 0 or b_gmean == 0:
                vals.append(np.nan)
                lo_err.append(0)
                hi_err.append(0)
            else:
                normed = c_vals[c_vals > 0] / b_gmean
                if len(normed) == 0:
                    vals.append(np.nan)
                    lo_err.append(0)
                    hi_err.append(0)
                    continue
                gm = np.exp(np.mean(np.log(normed)))
                q25 = np.percentile(normed, 25)
                q75 = np.percentile(normed, 75)
                vals.append(gm)
                lo_err.append(max(0, gm - q25))
                hi_err.append(max(0, q75 - gm))
        offset = (j - (n_comp - 1) / 2) * w
        ax.bar(group_x + offset, vals, w * 0.9,
               color=_color(c), edgecolor="black", linewidth=0.4,
               hatch=_hatch(c), label=_label(c),
               yerr=[lo_err, hi_err], error_kw=dict(lw=0.8))

    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xticks(group_x)
    ax.set_xticklabels([TEST_LABELS.get(t, t) for t, _ in tests],
                       rotation=30, ha="right")
    ax.set_ylim(bottom=0, top=3.0)
    # Label bars that exceed the cap with their value.
    for bar in ax.patches:
        h = bar.get_height()
        if h > 3.0:
            ax.text(bar.get_x() + bar.get_width() / 2, 2.95,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=5,
                    fontweight="bold", rotation=90,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
                    clip_on=True)


def plot2_normalized_throughput(df, base):
    d = df[df["nr_readers"] == 32]
    ALL = sorted(df['test'].unique())
    PLOT2_TESTS = [(t, int(df[df['test'] == t]['batch'].mode().iloc[0]))
                   for t in ALL]
    tests = PLOT2_TESTS
    if not tests:
        return

    cpu_counts = sorted(d["n_cpus"].unique())
    opts = sorted(d["opt"].dropna().unique())
    opt_compilers = {opt: _compilers_for_opt(df, opt, include_clang=False)
                     for opt in opts}
    opts = [o for o in opts if opt_compilers[o]]
    if not opts:
        return

    nrows = len(cpu_counts)
    ncols = len(opts)
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_W, 1.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for i, nc in enumerate(cpu_counts):
        for j, opt in enumerate(opts):
            ax = axes[i, j]
            _plot_normalized_bars(ax, d[(d["n_cpus"] == nc) & (d["opt"] == opt)],
                                  tests, opt_compilers[opt])
            title = opt if nrows == 1 else (f"{opt} — {nc} CPUs" if i == 0 else f"{nc} CPUs")
            ax.set_title(title, fontsize=8, fontweight="bold")
            if j == 0:
                ax.set_ylabel("Norm. reads")
            ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
            sns.despine(ax=ax)

    fig.suptitle("Normalized read throughput (32r/32w)",
                 fontsize=9, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.8)
    _add_legend(fig, list(axes[0]), y=0.96)
    _save(fig, base, "2_normalized")


def plot6_normalized_write_throughput(df, base):
    d = df[df["nr_readers"] == 32]
    ALL = sorted(df['test'].unique())
    PLOT6_TESTS = [(t, int(df[df['test'] == t]['batch'].mode().iloc[0]))
                   for t in ALL]
    tests = PLOT6_TESTS
    if not tests:
        return

    cpu_counts = sorted(d["n_cpus"].unique())
    opts = sorted(d["opt"].dropna().unique())
    opt_compilers = {opt: _compilers_for_opt(df, opt, include_clang=False)
                     for opt in opts}
    opts = [o for o in opts if opt_compilers[o]]
    if not opts:
        return

    nrows = len(cpu_counts)
    ncols = len(opts)
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_W, 1.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for i, nc in enumerate(cpu_counts):
        for j, opt in enumerate(opts):
            ax = axes[i, j]
            _plot_normalized_bars(ax, d[(d["n_cpus"] == nc) & (d["opt"] == opt)],
                                  tests, opt_compilers[opt],
                                  metric="writes_per_sec")
            title = opt if nrows == 1 else (f"{opt} — {nc} CPUs" if i == 0 else f"{nc} CPUs")
            ax.set_title(title, fontsize=8, fontweight="bold")
            if j == 0:
                ax.set_ylabel("Norm. writes")
            ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
            sns.despine(ax=ax)

    fig.suptitle("Normalized write throughput (32r/32w)",
                 fontsize=9, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.8)
    _add_legend(fig, list(axes[0]), y=0.96)
    _save(fig, base, "6_write_normalized")


# ---------------------------------------------------------------------------
# Plot 3: Reader scalability (line plot, O0/O3 rows)
# ---------------------------------------------------------------------------

PLOT3_TESTS = ["mutex", "urcu", "urcu_mb", "urcu_qsbr",
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
                    median="median",
                    q25=lambda x: x.quantile(0.25),
                    q75=lambda x: x.quantile(0.75)).reset_index()
                ax.plot(agg["nr_readers"], agg["median"],
                        label=_label(c), color=_color(c))
                ax.fill_between(agg["nr_readers"],
                                agg["q25"], agg["q75"],
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
    _add_legend(fig, all_axes, y=0.96)
    _save(fig, base, "3_reader_scalability")


# ---------------------------------------------------------------------------
# Plot 4: Hashtable normalized ops/s (grouped bars, O0/O3 rows)
# ---------------------------------------------------------------------------

PLOT4_TESTS = [("urcu_hash", 0)]


def plot4_hashtable(df, base):
    d = df[df["test"] == "urcu_hash"]
    if d.empty:
        return

    configs = sorted(d[["nr_readers", "nr_writers"]].drop_duplicates().values.tolist())
    tests = [(f"urcu_hash", 0)] * len(configs)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(SINGLE_W, 3.0), sharey=True)

    for ax, opt in [(ax_top, "O0"), (ax_bot, "O3")]:
        compilers = _compilers_for_opt(df, opt)
        d_opt = d[d["opt"] == opt]
        if d_opt.empty or not compilers:
            continue

        baseline_name = None
        for cand in [f"clangir-{opt}", f"clang-{opt}"]:
            if cand in d_opt["compiler"].unique():
                baseline_name = cand
                break
        if baseline_name is None:
            orb = [c for c in compilers if "fc" in c]
            baseline_name = min(orb, key=lambda c: int(c.split("fc")[-1])) if orb else compilers[0]

        n_comp = len(compilers)
        w = 0.8 / n_comp
        group_x = np.arange(len(configs))

        for j, c in enumerate(compilers):
            vals = []
            lo_err = []
            hi_err = []
            for nr, nw in configs:
                td = d_opt[(d_opt["nr_readers"] == nr) & (d_opt["nr_writers"] == nw)]
                c_vals = td[td["compiler"] == c]["ops_per_sec"].values
                b_vals = td[td["compiler"] == baseline_name]["ops_per_sec"].values
                b_gmean = np.exp(np.mean(np.log(b_vals[b_vals > 0]))) if np.any(b_vals > 0) else 0
                if len(c_vals) == 0 or b_gmean == 0:
                    vals.append(np.nan)
                    lo_err.append(0)
                    hi_err.append(0)
                else:
                    normed = c_vals[c_vals > 0] / b_gmean
                    if len(normed) == 0:
                        vals.append(np.nan)
                        lo_err.append(0)
                        hi_err.append(0)
                        continue
                    gm = np.exp(np.mean(np.log(normed)))
                    q25 = np.percentile(normed, 25)
                    q75 = np.percentile(normed, 75)
                    vals.append(gm)
                    lo_err.append(gm - q25)
                    hi_err.append(q75 - gm)
            offset = (j - (n_comp - 1) / 2) * w
            ax.bar(group_x + offset, vals, w * 0.9,
                   color=_color(c), edgecolor="black", linewidth=0.4,
                   hatch=_hatch(c), label=_label(c),
                   yerr=[lo_err, hi_err], error_kw=dict(lw=0.8))

        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_xticks(group_x)
        ax.set_xticklabels([f"{nr}r/{nw}w" for nr, nw in configs])
        ax.set_ylim(bottom=0)
        ax.set_title(opt, fontsize=8, fontweight="bold")
        ax.set_ylabel("Norm. ops")
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        sns.despine(ax=ax)

    fig.suptitle("Normalized hashtable throughput",
                 fontsize=9, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.8)
    _add_legend(fig, [ax_top, ax_bot], y=0.97)
    _save(fig, base, "4_hashtable")


# ---------------------------------------------------------------------------
# Plot 5: Synthesis convergence from nix build logs
# ---------------------------------------------------------------------------

# Compilation units to plot.  Keyed by source filename prefix; the
# per-module log files in $out/synth/ are named <source>:<hash>.log.
# When multiple logs match (same source, different defines), the longest is used.
_SYNTH_LABEL = {
    "test_mutex.c": "mutex", "test_rwlock.c": "rwlock",
    "test_perthreadlock.c": "ptlock",
    "test_urcu.c": "urcu", "test_urcu_qsbr.c": "qsbr",
    "test_urcu_gc.c": "gc", "test_urcu_qsbr_gc.c": "qsbr_gc",
}


def _synth_label(key):
    """Derive a short label from a synth key.

    New layout keys are target names like 'test_urcu_qsbr_gc'.
    Old layout keys are 'source.c:hash'.
    """
    if ":" in key:
        # Old flat layout: test_urcu_gc.c:2f8c
        src = key.split(":")[0]
        hsh = key.split(":")[1]
        base = _SYNTH_LABEL.get(src, src.removeprefix("test_").removesuffix(".c"))
        return f"{base}:{hsh}"
    # New per-target layout: test_urcu_qsbr_gc
    name = key.removeprefix("test_")
    return TEST_LABELS.get(name, name)

_ORDERED_RE = re.compile(
    r"ordered=(\d+)/(\d+)\s+overspecified=(\d+)")
_DONE_RE = re.compile(
    r"done ordered=(\d+)/(\d+)\s+overspecified=(\d+)")
_START_RE = re.compile(r"start fenceCostBase=")
_MATRIX_RE = re.compile(r"n=(\d+)\s+unreachable=(\d+)\s+reachable=(\d+)")
def _parse_one_synth_log(path):
    """Parse a single synthesis log file.

    Returns (reachable, [(step, ordered, total, overspec), ...]) or None.
    """
    steps = []
    step = 0
    reachable = None
    with open(path) as f:
        for line in f:
            if _START_RE.search(line):
                steps = []
                step = 0
                reachable = None
                continue
            m = _MATRIX_RE.search(line)
            if m:
                reachable = int(m.group(3))
                continue
            m = _ORDERED_RE.search(line)
            if not m:
                m = _DONE_RE.search(line)
            if m:
                ordered, total = int(m.group(1)), int(m.group(2))
                overspec = int(m.group(3))
                steps.append((step, ordered, total, overspec))
                step += 1
    return (reachable, steps) if steps else None


def _normalize_trace(reachable, raw):
    """Convert raw steps to [(step, remaining_frac, overspec_pct), ...]."""
    if not raw:
        return []
    total_required = raw[0][2]  # total from ordered=X/Y
    if total_required == 0:
        return []
    overspec_denom = total_required
    processed = []
    for step, ordered, _, overspec in raw:
        remaining = (total_required - ordered) / total_required
        remaining = max(remaining, 0.0)
        pct = overspec / overspec_denom * 100 if overspec_denom > 0 else 0
        processed.append((step, remaining, pct))
    return processed


def _parse_synth_log(result_dir):
    """Parse per-object synthesis logs from $result/synth/<target>/*.log.

    Returns dict: {target: trace}
    where trace = [(iteration, remaining_frac, overspec_pct), ...]

    Supports both new layout (synth/<target>/*.log) and old flat layout
    (synth/*.log).
    """
    synth_dir = os.path.join(os.path.realpath(result_dir), "synth")
    if not os.path.isdir(synth_dir):
        return {}

    result = {}

    # Check for new per-target subdirectory layout.
    subdirs = [d for d in sorted(os.listdir(synth_dir))
               if os.path.isdir(os.path.join(synth_dir, d))]
    if subdirs:
        for target in subdirs:
            target_dir = os.path.join(synth_dir, target)
            # Pick the longest trace among all logs for this target.
            best = None
            for logfile in sorted(os.listdir(target_dir)):
                if not logfile.endswith(".log"):
                    continue
                parsed = _parse_one_synth_log(os.path.join(target_dir, logfile))
                if parsed and (best is None or len(parsed[1]) > len(best[1])):
                    best = parsed
            if best:
                trace = _normalize_trace(best[0], best[1])
                if trace:
                    result[target] = trace
        return result

    # Fallback: old flat layout (synth/<source>:<hash>.log).
    raw_traces = {}
    for logfile in sorted(os.listdir(synth_dir)):
        if not logfile.endswith(".log"):
            continue
        obj_name = logfile[:-4]
        parsed = _parse_one_synth_log(os.path.join(synth_dir, logfile))
        if parsed:
            raw_traces[obj_name] = parsed

    for f, (reachable, raw) in raw_traces.items():
        trace = _normalize_trace(reachable, raw)
        if trace:
            result[f] = trace
    return result


_NAIVE_DONE_RE = re.compile(
    r"done ordered=(\d+)/(\d+)\s+overspecified=(\d+)")
_NAIVE_N_RE = re.compile(r"\bn=(\d+)\b")


def _parse_one_naive_log(path):
    """Parse a single naive synth log. Returns (overspec, total) or None."""
    overspec = None
    total = None
    with open(path) as f:
        for line in f:
            m = _NAIVE_DONE_RE.search(line)
            if m:
                overspec = int(m.group(3))
                total = int(m.group(2))
    if overspec is not None and total and total > 0:
        return (overspec, total)
    return None


def _parse_naive_overspec(result_dir):
    """Parse naive synth logs for final overspecified % per target.

    Returns dict: {target: overspec_pct}
    """
    synth_dir = os.path.join(os.path.realpath(result_dir), "synth")
    if not os.path.isdir(synth_dir):
        return {}
    result = {}

    # New per-target subdirectory layout.
    subdirs = [d for d in sorted(os.listdir(synth_dir))
               if os.path.isdir(os.path.join(synth_dir, d))]
    if subdirs:
        for target in subdirs:
            target_dir = os.path.join(synth_dir, target)
            # Pick the log with the highest overspec (most interesting).
            best_pct = None
            for logfile in sorted(os.listdir(target_dir)):
                if not logfile.endswith(".log"):
                    continue
                parsed = _parse_one_naive_log(os.path.join(target_dir, logfile))
                if parsed:
                    pct = parsed[0] / parsed[1] * 100
                    if best_pct is None or pct > best_pct:
                        best_pct = pct
            if best_pct is not None:
                result[target] = best_pct
        return result

    # Fallback: old flat layout.
    for logfile in sorted(os.listdir(synth_dir)):
        if not logfile.endswith(".log"):
            continue
        obj_name = logfile[:-4]
        parsed = _parse_one_naive_log(os.path.join(synth_dir, logfile))
        if parsed:
            pct = parsed[0] / parsed[1] * 100
            result[obj_name] = pct
    return result


def plot5_synthesis(config_path, base, opt="O0"):
    """Plot synthesis convergence curves from nix build logs."""
    if not config_path or not os.path.isfile(config_path):
        return
    with open(config_path) as f:
        cfg = json.load(f)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    # Collect orb compilers for the requested opt level with fc in FC_SIZES.
    orb_compilers = []
    for c in cfg["compilers"]:
        name = c["name"]
        if not name.startswith(f"orb-{opt}"):
            continue
        result_dir = os.path.join(config_dir, f"result-{name}")
        if not os.path.exists(result_dir):
            continue
        m = re.search(r"fc(\d+)", name)
        if not m or int(m.group(1)) not in FC_SIZES:
            continue
        orb_compilers.append((name, result_dir))

    # Look for naive result directory (may not be in config).
    naive_dir = None
    for candidate in [f"result-orb-{opt}-naive", f"result-naive-{opt}"]:
        p = os.path.join(config_dir, candidate)
        if os.path.isdir(os.path.join(os.path.realpath(p), "synth") if os.path.exists(p) else ""):
            naive_dir = p
            break

    if not orb_compilers:
        return

    # Parse logs for each compiler.
    all_data = {}  # {compiler_name: {source_file: trace}}
    for name, rdir in orb_compilers:
        try:
            all_data[name] = _parse_synth_log(rdir)
        except Exception as e:
            print(f"warning: could not parse log for {name}: {e}")

    # Parse naive overspecification for reference lines.
    naive_overspec = {}
    if naive_dir:
        try:
            naive_overspec = _parse_naive_overspec(naive_dir)
        except Exception as e:
            print(f"warning: could not parse naive logs: {e}")

    # Auto-discover synth log keys, keeping only targets that appear in
    # TEST_LABELS (i.e. the benchmarks shown in normalized plots).
    # Keys are target names like "test_urcu_qsbr_gc" (new layout) or
    # "test_urcu_gc.c:2f8c" (old flat layout).
    all_keys = set()
    for data in all_data.values():
        for k in data:
            # New layout: key = "test_<name>", check <name> in TEST_LABELS
            # Old layout: key = "test_<name>.c:<hash>", check <name> in TEST_LABELS
            name = k.split(":")[0].removeprefix("test_").removesuffix(".c")
            if name in TEST_LABELS:
                all_keys.add(k)
    files = sorted(all_keys)
    if not files:
        return

    ncols = len(files)
    fig, axes = plt.subplots(2, ncols, figsize=(max(DOUBLE_W, ncols * 0.9), 3.2),
                             squeeze=False, sharex=True, sharey='row')

    for i, key in enumerate(files):
        ax_top = axes[0][i]
        ax_bot = axes[1][i]
        dep_frac = None
        for name, _ in orb_compilers:
            trace = all_data.get(name, {}).get(key, [])
            if not trace:
                continue
            xs = [t[0] for t in trace]
            remaining = [t[1] * 100 for t in trace]
            overspec = [t[2] for t in trace]
            if dep_frac is None:
                dep_frac = 100 - remaining[0]
            lbl = _label(name)
            ax_top.plot(xs, remaining, label=lbl, color=_color(name),
                        linewidth=1.2, marker=None)
            ax_bot.plot(xs, overspec, label=lbl, color=_color(name),
                        linewidth=1.2, marker=None)
        # Highlight initial dependency drop.
        if dep_frac is not None:
            ax_top.axhspan(100 - dep_frac, 100, color="gray", alpha=0.12)
            ax_top.text(0.96, 0.96, f"{dep_frac:.0f}%\ndeps.",
                        transform=ax_top.transAxes, fontsize=5.5,
                        ha="right", va="top", color="0.4")
        # Draw naive overspecification reference line.
        naive_pct = naive_overspec.get(key)
        if naive_pct is not None:
            ax_bot.axhline(naive_pct, color="0.4", linewidth=0.8,
                           linestyle=":", zorder=1)
            if i == len(files) - 1:
                ax_bot.text(1.02, naive_pct, "naive",
                            transform=ax_bot.get_yaxis_transform(),
                            fontsize=5.5, va="center", color="0.4")
        ax_top.set_title(_synth_label(key), fontsize=7, fontweight="bold")
        ax_top.set_ylim(-2, 102)
        ax_top.grid(True, alpha=0.3, linewidth=0.5)
        ax_bot.grid(True, alpha=0.3, linewidth=0.5)
        ax_bot.set_xlabel("")
        sns.despine(ax=ax_top)
        sns.despine(ax=ax_bot)
        if i == 0:
            ax_top.set_ylabel("% unordered")
            ax_bot.set_ylabel("% over-ordered\n(of required)")

    fig.suptitle("Synthesis convergence",
                 fontsize=9, fontweight="bold", y=0.99)
    fig.supxlabel("Synthesis iterations")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93], w_pad=0.8, h_pad=0.6)
    _add_legend(fig, list(axes[0]), y=0.94)
    _save(fig, base, "5_synthesis")


# ---------------------------------------------------------------------------
# Plot 7: Compile time vs. number of memory events
# ---------------------------------------------------------------------------

def _parse_cc_times(result_dir):
    """Parse cc-times/times.log from a nix build result.

    Supports both old format (source ms) and new format (source object ms).
    Returns dict: {source_basename: [elapsed_ms, ...]}
    """
    times_path = os.path.join(os.path.realpath(result_dir), "cc-times", "times.log")
    if not os.path.isfile(times_path):
        return {}
    result = {}
    with open(times_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                src, ms = parts[0], int(parts[1])
            elif len(parts) == 3:
                src, _obj, ms = parts[0], parts[1], int(parts[2])
            else:
                continue
            result.setdefault(src, []).append(ms)
    return result


_REQUIRED_PAIRS_RE = re.compile(r"required pairs=(\d+)")


def _parse_synth_n(result_dir):
    """Extract n (event count) and required pairs from each synth log.

    Returns (ns, pairs) where both are dict: {source_basename: [val, ...]}
    sorted ascending by n.
    """
    synth_dir = os.path.join(os.path.realpath(result_dir), "synth")
    if not os.path.isdir(synth_dir):
        return {}, {}

    ns = {}
    pairs = {}

    def _scan_log(path, src):
        n_val = None
        p_val = None
        with open(path) as f:
            for line in f:
                if n_val is None:
                    m = _MATRIX_RE.search(line)
                    if m:
                        n_val = int(m.group(1))
                    else:
                        m2 = re.search(r"\bn=(\d+)\b", line)
                        if m2:
                            n_val = int(m2.group(1))
                if p_val is None:
                    m = _REQUIRED_PAIRS_RE.search(line)
                    if m:
                        p_val = int(m.group(1))
                if n_val is not None and p_val is not None:
                    break
        if n_val is not None:
            ns.setdefault(src, []).append(n_val)
        if p_val is not None:
            pairs.setdefault(src, []).append(p_val)

    subdirs = [d for d in os.listdir(synth_dir)
               if os.path.isdir(os.path.join(synth_dir, d))]
    if subdirs:
        for target in subdirs:
            target_dir = os.path.join(synth_dir, target)
            for logfile in os.listdir(target_dir):
                if logfile.endswith(".log"):
                    _scan_log(os.path.join(target_dir, logfile),
                              logfile.split(":")[0])
    else:
        for logfile in os.listdir(synth_dir):
            if logfile.endswith(".log"):
                _scan_log(os.path.join(synth_dir, logfile),
                          logfile.split(":")[0])

    for src in ns:
        ns[src].sort()
    for src in pairs:
        pairs[src].sort()
    return ns, pairs


def plot7_compile_time_vs_n(config_path, base):
    """Line plot: n (memory events) vs compile time (ms), O0/O3 columns."""
    if not config_path or not os.path.isfile(config_path):
        return
    with open(config_path) as f:
        cfg = json.load(f)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    # Collect clangir + orb-fc compilers, grouped by opt level.
    by_opt = {}  # {opt: [(name, result_dir), ...]}
    for c in cfg["compilers"]:
        name = c["name"]
        if "naive" in name or name.startswith("clang-"):
            continue
        result_dir = os.path.join(config_dir, f"result-{name}")
        if not os.path.exists(result_dir):
            continue
        m = re.search(r"-(O\d)", name)
        if not m:
            continue
        opt = m.group(1)
        by_opt.setdefault(opt, []).append((name, result_dir))

    opts = sorted(by_opt)
    if not opts:
        return

    # Build reference n and pairs mappings from the first orb compiler.
    ref_ns = {}
    ref_pairs = {}
    for opt_compilers in by_opt.values():
        for name, rdir in opt_compilers:
            ns, pairs = _parse_synth_n(rdir)
            if ns:
                ref_ns = ns
                ref_pairs = pairs
                break
        if ref_ns:
            break

    fig, axes = plt.subplots(2, len(opts), figsize=(SINGLE_W, 2.0),
                             sharex=False, sharey=True, squeeze=False)

    for j, opt in enumerate(opts):
        ax_time = axes[0][j]
        ax_pairs = axes[1][j]
        for name, rdir in by_opt[opt]:
            cc_times = _parse_cc_times(rdir)
            synth_ns, synth_pairs = _parse_synth_n(rdir)
            if not synth_ns:
                synth_ns = ref_ns
                synth_pairs = ref_pairs

            # Collect (n, time) and (n, pairs) points.
            points_n = []
            points_t = []
            points_p_n = []
            points_p = []
            for src, times_list in cc_times.items():
                if not src.startswith("test_"):
                    continue
                if src in synth_ns:
                    ns = synth_ns[src]
                    ts = sorted(times_list)
                    for n, t in zip(ns, ts):
                        points_n.append(n)
                        points_t.append(t)
                if src in synth_pairs:
                    ps = synth_pairs[src]
                    ts = sorted(times_list)
                    for p, t in zip(ps, ts):
                        points_p_n.append(p)
                        points_p.append(t)

            col = _color(name)
            lbl = _label(name)

            # Top row: compile time vs n
            if points_n:
                by_n = {}
                for n, t in zip(points_n, points_t):
                    by_n.setdefault(n, []).append(t)
                ns_sorted = sorted(by_n)
                medians = [np.median(by_n[n]) for n in ns_sorted]
                q25s = [np.percentile(by_n[n], 25) for n in ns_sorted]
                q75s = [np.percentile(by_n[n], 75) for n in ns_sorted]
                ax_time.plot(ns_sorted, medians, label=lbl, color=col,
                             linewidth=1.2)
                ax_time.fill_between(ns_sorted, q25s, q75s,
                                     color=col, alpha=0.15)

            # Bottom row: compile time vs required pairs
            if points_p_n:
                by_p = {}
                for p, t in zip(points_p_n, points_p):
                    by_p.setdefault(p, []).append(t)
                ps_sorted = sorted(by_p)
                medians = [np.median(by_p[p]) for p in ps_sorted]
                q25s = [np.percentile(by_p[p], 25) for p in ps_sorted]
                q75s = [np.percentile(by_p[p], 75) for p in ps_sorted]
                ax_pairs.plot(ps_sorted, medians, label=lbl, color=col,
                              linewidth=1.2)
                ax_pairs.fill_between(ps_sorted, q25s, q75s,
                                      color=col, alpha=0.15)

        ax_time.set_title(opt, fontsize=8, fontweight="bold")
        ax_time.grid(True, alpha=0.3, linewidth=0.5)
        ax_pairs.grid(True, alpha=0.3, linewidth=0.5)
        ax_pairs.set_xlabel("Required pairs")
        ax_time.set_xlabel("Memory events (n)")
        sns.despine(ax=ax_time)
        sns.despine(ax=ax_pairs)

    fig.supylabel("Compile time (ms)", fontsize=8)

    fig.suptitle("Compile time vs. memory events / required pairs",
                 fontsize=9, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=0.8)
    _add_legend(fig, list(axes[0]), y=0.95)
    _save(fig, base, "7_compile_time")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--config", default=None,
                    help="JSON config for synthesis plot (nix log parsing)")
    ap.add_argument("--opt", default="O0",
                    help="Opt level for synthesis plot (default: O0)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    df = load_runs(Path(args.rundir))
    print(f"Loaded {len(df)} data points from {args.rundir}")
    print(f"Compilers: {sorted(df['compiler'].unique())}")
    print(f"Tests: {sorted(df['test'].unique())}")

    for fn in [plot1_throughput_vs_batch, plot2_normalized_throughput,
                plot3_reader_scalability, plot4_hashtable,
                plot6_normalized_write_throughput]:
        try:
            fn(df, args.output)
        except Exception as e:
            print(f"Skipping {fn.__name__}: {e}")
    plot5_synthesis(args.config, args.output, args.opt)
    plot7_compile_time_vs_n(args.config, args.output)

    if args.show:
        plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
