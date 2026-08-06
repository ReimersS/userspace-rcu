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

FC_SIZES = {1: "L", 333: "H"}

LABEL_STYLE = {
    "clangir": (PAL[1], HATCHES[1]),
    "clang":   (PAL[0], HATCHES[0]),
    "orb L":   (PAL[3], HATCHES[3]),
    "orb H":   (PAL[2], HATCHES[2]),
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
                dur = int(m.group(2))
                reads, writes, ops = int(m.group(8)), int(m.group(9)), int(m.group(10))
                rows.append({
                    "compiler": compiler, "test": m.group(1), "run": run,
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
            "compiler", "test", "run", "duration", "nr_readers", "nr_writers",
            "rdur", "wdur", "wdelay", "batch", "reads", "writes", "ops",
            "reads_per_sec", "writes_per_sec", "ops_per_sec", "opt", "fc"])
        return df
    df = df.drop_duplicates(
        subset=["compiler", "test", "run", "batch", "nr_readers",
                "nr_writers", "wdelay", "rdur", "wdur"],
        keep="first")
    df["opt"] = df["compiler"].str.extract(r"-(O\d)")
    df["fc"] = df["compiler"].str.extract(r"-fc(\d+)").astype(float)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compilers_for_opt(df, opt):
    present = set(df[df["opt"] == opt]["compiler"].unique())
    order = []
    # clangir first (baseline), then clang, then orb variants
    for prefix in [f"clangir-{opt}", f"clang-{opt}"]:
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

PLOT2_TESTS = [
    ("mutex", 0), ("rwlock", 0), ("urcu", 0), ("urcu_mb", 0),
    ("urcu_qsbr", 0), ("urcu_gc", 1), ("urcu_mb_gc", 32768),
    ("urcu_qsbr_gc", 32768),
]


def _plot_normalized_bars(ax, df_opt, tests, compilers, metric="reads_per_sec"):
    if not compilers:
        return
    opt = compilers[0].split("-")[1]
    baseline = None
    for cand in [f"clangir-{opt}", f"clang-{opt}"]:
        if cand in df_opt["compiler"].unique():
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
        for test, batch in tests:
            td = df_opt[(df_opt["test"] == test) & (df_opt["batch"] == batch)]
            c_mean = td[td["compiler"] == c][metric].mean()
            b_mean = td[td["compiler"] == baseline][metric].mean()
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

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(DOUBLE_W, 3.0),sharex=True, sharey=True)

    for ax, opt in [(ax_top, "O0"), (ax_bot, "O3")]:
        _plot_normalized_bars(ax, d[d["opt"] == opt], tests,
                              _compilers_for_opt(df, opt))
        ax.set_title(opt, fontsize=8, fontweight="bold")
        ax.set_ylabel("Norm. reads")
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        sns.despine(ax=ax)

    fig.suptitle("Normalized read throughput (32r/32w)",
                 fontsize=9, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.8)
    _add_legend(fig, [ax_top, ax_bot], y=0.96)
    _save(fig, base, "2_normalized")


# ---------------------------------------------------------------------------
# Plot 3: Reader scalability (line plot, O0/O3 rows)
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
            for nr, nw in configs:
                td = d_opt[(d_opt["nr_readers"] == nr) & (d_opt["nr_writers"] == nw)]
                c_mean = td[td["compiler"] == c]["ops_per_sec"].mean()
                b_mean = td[td["compiler"] == baseline_name]["ops_per_sec"].mean()
                vals.append(c_mean / b_mean if pd.notna(c_mean) and b_mean > 0 else np.nan)
            offset = (j - (n_comp - 1) / 2) * w
            ax.bar(group_x + offset, vals, w * 0.9,
                   color=_color(c), edgecolor="black", linewidth=0.4,
                   hatch=_hatch(c), label=_label(c))

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
SYNTH_FILES = [
    ("test_mutex.c", "mutex"),
    ("test_rwlock.c", "rwlock"),
    ("test_urcu.c", "urcu"),
    ("test_urcu_qsbr.c", "qsbr"),
    ("test_urcu_gc.c", "gc"),
    ("test_urcu_qsbr_gc.c", "qsbr_gc"),
]

_ORDERED_RE = re.compile(
    r"ordered=(\d+)/(\d+)\s+overspecified=(\d+)")
_DONE_RE = re.compile(
    r"done ordered=(\d+)/(\d+)\s+overspecified=(\d+)")
_START_RE = re.compile(r"start fenceCostBase=")
_MATRIX_RE = re.compile(r"n=(\d+)\s+unreachable=(\d+)\s+reachable=(\d+)")


def _parse_synth_log(result_dir):
    """Parse per-object synthesis logs from $result/synth/*.log.

    Returns dict: {obj_filename: [(iteration, remaining_frac, overspec_pct), ...]}
    where remaining_frac is normalized to the effective maximum (final ordered count,
    excluding unreachable pairs) and overspec_pct = overspecified / reachable * 100.
    """
    synth_dir = os.path.join(os.path.realpath(result_dir), "synth")
    if not os.path.isdir(synth_dir):
        return {}

    raw_traces = {}  # {obj_name: (reachable, [(step, ordered, total, overspec), ...])}
    for logfile in sorted(os.listdir(synth_dir)):
        if not logfile.endswith(".log"):
            continue
        obj_name = logfile[:-4]
        path = os.path.join(synth_dir, logfile)
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
        if steps:
            raw_traces[obj_name] = (reachable, steps)

    # Post-process: normalize remaining by effective max (final ordered count),
    # overspecified as % of reachable pairs.
    result = {}
    for f, (reachable, raw) in raw_traces.items():
        if not raw:
            continue
        effective_max = raw[-1][1]  # final ordered count
        if effective_max == 0:
            effective_max = raw[0][2]  # fallback to total
        # Denominator for overspecified %: reachable if available, else n²-n estimate.
        overspec_denom = reachable if reachable else raw[0][2]
        processed = []
        for step, ordered, _, overspec in raw:
            remaining = (effective_max - ordered) / effective_max if effective_max > 0 else 0
            remaining = max(remaining, 0.0)
            pct = overspec / overspec_denom * 100 if overspec_denom > 0 else 0
            processed.append((step, remaining, pct))
        result[f] = processed
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
        m = re.search(r"fc(\d+)", name)
        if not m or int(m.group(1)) not in FC_SIZES:
            continue
        result_dir = os.path.join(config_dir, f"result-{name}")
        if os.path.exists(result_dir):
            orb_compilers.append((name, result_dir))

    if not orb_compilers:
        return

    # Parse logs for each compiler.
    all_data = {}  # {compiler_name: {source_file: trace}}
    for name, rdir in orb_compilers:
        try:
            all_data[name] = _parse_synth_log(rdir)
        except Exception as e:
            print(f"warning: could not parse log for {name}: {e}")

    files = SYNTH_FILES

    ncols = len(files)
    fig, axes = plt.subplots(2, ncols, figsize=(DOUBLE_W, 3.2),
                             squeeze=False, sharex=True, sharey='row')

    def _lookup_trace(data, prefix):
        """Find longest trace whose key starts with prefix (handles :hash suffix)."""
        matches = [(k, v) for k, v in data.items() if k.startswith(prefix)]
        if not matches:
            return []
        return max(matches, key=lambda kv: len(kv[1]))[1]

    for i, (src, label) in enumerate(files):
        ax_top = axes[0][i]
        ax_bot = axes[1][i]
        dep_frac = None  # fraction resolved by dependencies (iteration 0)
        for name, _ in orb_compilers:
            trace = _lookup_trace(all_data.get(name, {}), src)
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
        ax_top.set_title(label, fontsize=8, fontweight="bold")
        ax_top.set_ylim(-2, 102)
        ax_top.grid(True, alpha=0.3, linewidth=0.5)
        ax_bot.grid(True, alpha=0.3, linewidth=0.5)
        ax_bot.set_xlabel("")
        sns.despine(ax=ax_top)
        sns.despine(ax=ax_bot)
        if i == 0:
            ax_top.set_ylabel("% unordered")
            ax_bot.set_ylabel("% over-ordered\n(of reachable)")

    fig.suptitle("Synthesis convergence",
                 fontsize=9, fontweight="bold", y=0.99)
    fig.supxlabel("Synthesis iterations")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93], w_pad=0.8, h_pad=0.6)
    _add_legend(fig, list(axes[0]), y=0.94)
    _save(fig, base, "5_synthesis")


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
                plot3_reader_scalability, plot4_hashtable]:
        try:
            fn(df, args.output)
        except Exception as e:
            print(f"Skipping {fn.__name__}: {e}")
    plot5_synthesis(args.config, args.output, args.opt)

    if args.show:
        plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
