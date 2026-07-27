#!/usr/bin/env python3
"""
Two figures:
  1. FC sweep   — orb variants at different fence costs, O0 and O3 rows.
  2. Best vs    — best orb FC compared against clang and clangir, O0 and O3 rows.
Each figure has one column per benchmark.
"""
import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("csv", nargs="?", default="runs/latest/results.csv")
parser.add_argument("--output", default=None,
                    help="output base path; -fc and -best suffixes are added")
args = parser.parse_args()

df = pd.read_csv(args.csv)

# Derive opt level (O0/O3) and variant type (orb/clangir/clang).
df["opt"]     = df["compiler"].str.extract(r"-(O\d)")
df["variant"] = df["compiler"].str.extract(r"^(orb|clangir|clang)")
df["fc"]      = df["compiler"].str.extract(r"-fc(\d+)").astype(float)

benchmarks = sorted(df["benchmark"].unique())
opt_levels = ["O0", "O3"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ax(axes, row, col, n_cols):
    """Return the correct axes object regardless of subplot shape."""
    if len(opt_levels) == 1 and n_cols == 1:
        return axes
    if len(opt_levels) == 1:
        return axes[col]
    if n_cols == 1:
        return axes[row]
    return axes[row][col]


def _normalize(data, group_cols, baseline_mask, y="throughput_ops_per_sec"):
    """Divide y by the per-group mean of rows selected by baseline_mask."""
    baseline = (
        data[baseline_mask]
        .groupby(group_cols)[y]
        .mean()
        .rename("_baseline")
    )
    data = data.join(baseline, on=group_cols)
    data["rel"] = data[y] / data["_baseline"]
    return data.drop(columns="_baseline")


def _barplot(ax, data, hue_col, y="throughput_ops_per_sec", ylabel="throughput (ops/s)",
             hue_order=None):
    order = sorted(data["test_program"].unique())
    sns.barplot(
        data=data,
        x="test_program",
        y=y,
        hue=hue_col,
        hue_order=hue_order,
        palette="pastel",
        order=order,
        ax=ax,
        errorbar="sd",
    )
    if y == "rel":
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=7, rotation=30)


def _finalize(fig, output):
    handles, labels = None, None
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend:
            legend.remove()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncols=len(labels), bbox_to_anchor=(0.5, 1.02), frameon=False)
    plt.tight_layout()
    if output:
        plt.savefig(output, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


def output_path(template, suffix):
    if template is None:
        return None
    base, _, ext = template.rpartition(".")
    return f"{base}-{suffix}.{ext}" if base else f"{template}-{suffix}"


# ---------------------------------------------------------------------------
# Figure 1: FC sweep (orb only)
# ---------------------------------------------------------------------------

def plot_fc_sweep():
    data = df[(df["variant"] == "orb") & df["fc"].notna()].copy()
    if data.empty:
        print("No orb-fc data found; skipping FC sweep plot.")
        return

    # Label: "fc=N" ordered numerically.
    data["label"] = "fc=" + data["fc"].astype(int).astype(str)
    fc_order = ["fc=" + str(int(v)) for v in sorted(data["fc"].unique())]

    # Normalize per (benchmark, opt, test_program) to the lowest fc.
    min_fc = data["fc"].min()
    group = ["benchmark", "opt", "test_program"]
    data = _normalize(data, group, data["fc"] == min_fc)
    baseline_label = f"fc={int(min_fc)}"

    fig, axes = plt.subplots(
        len(opt_levels), len(benchmarks),
        figsize=(7 * len(benchmarks), 5 * len(opt_levels)),
        squeeze=False,
    )
    fig.suptitle("Orb fence-cost sweep", y=1.03)

    for row, opt in enumerate(opt_levels):
        for col, bname in enumerate(benchmarks):
            ax = axes[row][col]
            sub = data[(data["benchmark"] == bname) & (data["opt"] == opt)]
            if sub.empty:
                ax.set_visible(False)
                continue
            _barplot(ax, sub, "label", y="rel",
                     ylabel=f"throughput relative to {baseline_label}",
                     hue_order=fc_order)
            ax.set_title(f"{bname} [{opt}]")

    _finalize(fig, output_path(args.output, "fc"))


# ---------------------------------------------------------------------------
# Figure 2: Best orb vs clang vs clangir
# ---------------------------------------------------------------------------

def find_best_orb_per_opt():
    """Return {opt: compiler_name} for the FC variant with highest mean throughput."""
    orb = df[df["variant"] == "orb"]
    best = {}
    for opt in opt_levels:
        sub = orb[orb["opt"] == opt]
        if sub.empty:
            continue
        means = sub.groupby("compiler")["throughput_ops_per_sec"].mean()
        best[opt] = means.idxmax()
    return best


def plot_best_vs():
    best_fc = find_best_orb_per_opt()
    if not best_fc:
        print("No orb-fc data found; skipping best-vs plot.")
        return

    # Build comparison dataset: best orb + clang + clangir, per opt.
    parts = []
    for opt, best_compiler in best_fc.items():
        fc_num = re.search(r"-fc(\d+)", best_compiler)
        orb_label = f"orb-fc{fc_num.group(1)}" if fc_num else "orb"

        orb_data = df[df["compiler"] == best_compiler].copy()
        orb_data["label"] = orb_label
        parts.append(orb_data)

        for variant, label in [("clang", "clang"), ("clangir", "clangir")]:
            vdata = df[(df["variant"] == variant) & (df["opt"] == opt)].copy()
            vdata["label"] = label
            parts.append(vdata)

    cmp = pd.concat(parts, ignore_index=True)

    # Normalize per (benchmark, opt, test_program) to clangir.
    group = ["benchmark", "opt", "test_program"]
    cmp = _normalize(cmp, group, cmp["label"] == "clangir")

    fig, axes = plt.subplots(
        len(opt_levels), len(benchmarks),
        figsize=(7 * len(benchmarks), 5 * len(opt_levels)),
        squeeze=False,
    )
    fig.suptitle("Best orb vs clang vs clangir", y=1.03)

    for row, opt in enumerate(opt_levels):
        for col, bname in enumerate(benchmarks):
            ax = axes[row][col]
            sub = cmp[(cmp["benchmark"] == bname) & (cmp["opt"] == opt)]
            if sub.empty:
                ax.set_visible(False)
                continue
            hue_order = sorted(sub["label"].unique())
            _barplot(ax, sub, "label", y="rel",
                     ylabel="throughput relative to clangir",
                     hue_order=hue_order)
            best_name = best_fc.get(opt, "")
            fc_tag = re.search(r"-fc(\d+)", best_name)
            title_tag = f" [best=fc{fc_tag.group(1)}]" if fc_tag else ""
            ax.set_title(f"{bname} [{opt}]{title_tag}")

    _finalize(fig, output_path(args.output, "best"))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

plot_fc_sweep()
plot_best_vs()
