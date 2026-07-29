#!/usr/bin/env python3
"""
Three figures:
  1. FC sweep      — orb variants at different fence costs, O0 and O3 rows.
  2. Best vs       — best orb FC compared against clang and clangir, O0 and O3 rows.
  3. Compile time  — total synthesis time vs fence cost, O0 and O3 rows, one line per benchmark.
Each figure has one column per benchmark (except Fig 3: one column total).
"""
import argparse
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

DISTANCE_MARKERS = {3: "o", 5: "s", 7: "^", 9: "D", 11: "P", 13: "X"}
# Label offset direction per distance so annotations don't collide
# (x_sign, y_sign) in data-space log-units; tuned for typical data layout
LABEL_OFFSETS = {3: (-1, 1), 5: (-1, 1), 7: (-1, 1), 9: (-1, 1), 11: (-1, 1), 13: (-1, 1)}

FONTSIZE = 12
tex_fonts = {
    # Use LaTeX to write all text
    # "text.usetex": True,
    "font.family": "serif",
    # Font sizes
    "axes.labelsize": FONTSIZE * 1.8,
    "font.size": FONTSIZE * 1.3,
    "legend.fontsize": (FONTSIZE - 2) * 1.5,
    "xtick.labelsize": (FONTSIZE - 1) * 1.5,
    "ytick.labelsize": (FONTSIZE - 1) * 1.5,
    "axes.titlesize": 12,
    # Line and marker styles
    "lines.linewidth": 1.5,
    "lines.markersize": 12,
    "lines.markeredgewidth": 1.5,
    "lines.markeredgecolor": "black",
    # Error bar cap size
    "errorbar.capsize": 3,
}
plt.rcParams.update(tex_fonts)

parser = argparse.ArgumentParser()
parser.add_argument("csv", nargs="?", default="runs/latest/results.csv")
parser.add_argument("--output", default=None,
                    help="output base path; -fc, -best, and -compile suffixes are added")
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


def _finalize(fig, output, legend_anchor=(0.5, 1.02), tight_rect=None):
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
                   ncols=len(labels), bbox_to_anchor=legend_anchor, frameon=False)
    plt.tight_layout(rect=tight_rect)
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
# Figure 3: Synthesis compile time per CU (bar), hue = fc
# ---------------------------------------------------------------------------

def _load_synth():
    synth_path = Path(args.csv).parent / "synthesis.csv"
    if synth_path.exists():
        s = pd.read_csv(synth_path)
    else:
        print("No synthesis.csv; trying nix log fallback.")
        s, _ = _load_nix_data()
        if s.empty:
            return None
    s["opt"] = s["compiler"].str.extract(r"-(O\d)")
    s["fc"]  = s["compiler"].str.extract(r"-fc(\d+)").astype(float)
    s["label"] = "fc=" + s["fc"].astype("Int64").astype(str)
    return s[s["fc"].notna()].copy()


def plot_compile_time():
    synth = _load_synth()
    if synth is None or synth.empty:
        print("No synthesis.csv / orb-fc data; skipping compile time plot.")
        return

    fc_order = ["fc=" + str(int(v)) for v in sorted(synth["fc"].unique())]

    fig, axes = plt.subplots(
        len(opt_levels), 1,
        figsize=(max(6, len(synth["cu_name"].unique()) * 1.2), 5 * len(opt_levels)),
        squeeze=False,
    )
    fig.suptitle("Synthesis time per compilation unit", y=1.03)

    for row, opt in enumerate(opt_levels):
        ax = axes[row][0]
        sub = synth[synth["opt"] == opt]
        if sub.empty:
            ax.set_visible(False)
            continue
        cu_order = sorted(sub["cu_name"].unique())
        sns.barplot(data=sub, x="cu_name", y="synthesis_time_ms",
                    hue="label", hue_order=fc_order,
                    order=cu_order, palette="pastel", ax=ax, errorbar="sd")
        ax.set_xlabel("compilation unit")
        ax.set_ylabel("synthesis time (ms)")
        ax.set_title(f"[{opt}]")
        ax.tick_params(axis="x", labelsize=7, rotation=30)

    _finalize(fig, output_path(args.output, "compile"))


# ---------------------------------------------------------------------------
# Nix log parsing (shared by Fig 3 and Fig 4)
# ---------------------------------------------------------------------------

_CC_STEP   = re.compile(r'\bCC\b\s+\S*?(?:la-)?(\w+)\.lo\b')
_REQUIRED  = re.compile(r'\[FenceSynthesis\] required pairs=(\d+)')
_ORDERED   = re.compile(r'\[FenceSynthesis\] (?:done )?ordered=(\d+)/\d+')
_DONE      = re.compile(r'\[FenceSynthesis\] done .* t=(\d+)ms')

_nix_cache: tuple[pd.DataFrame, pd.DataFrame] | None = None


def _parse_nix_log(log_text: str, compiler: str) -> tuple[list[dict], list[dict]]:
    """Return (synth_rows, conv_rows) parsed from one build log."""
    synth, conv = [], []
    cu_name = source = None
    iteration = 0
    for line in log_text.splitlines():
        m = _CC_STEP.search(line)
        if m:
            cu_name, source, iteration = m.group(1), None, 0
            continue
        m = _REQUIRED.search(line)
        if m:
            source, iteration = int(m.group(1)), 0
            continue
        m = _DONE.search(line)
        if m and cu_name:
            synth.append({"compiler": compiler, "cu_name": cu_name,
                          "synthesis_time_ms": int(m.group(1))})
        m = _ORDERED.search(line)
        if m and cu_name and source is not None:
            conv.append({"compiler": compiler, "cu_name": cu_name,
                         "iteration": iteration,
                         "outstanding": source - int(m.group(1)),
                         "total": source})
            iteration += 1
    return synth, conv


def _load_nix_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and cache (synth_df, conv_df) from result-orb-*-fc* nix logs."""
    global _nix_cache
    if _nix_cache is not None:
        return _nix_cache
    links = sorted(Path(".").resolve().glob("result-orb-*-fc*"))
    all_synth, all_conv = [], []
    for link in links:
        compiler = link.name.removeprefix("result-")
        try:
            result = subprocess.run(["nix", "log", str(link)],
                                    capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[warn] nix log failed for {link}: {e}")
            continue
        s, c = _parse_nix_log(result.stdout, compiler)
        all_synth.extend(s)
        all_conv.extend(c)
    synth_df = pd.DataFrame(all_synth) if all_synth else pd.DataFrame()
    conv_df  = pd.DataFrame(all_conv)  if all_conv  else pd.DataFrame()
    _nix_cache = (synth_df, conv_df)
    return _nix_cache


# ---------------------------------------------------------------------------
# Figure 4: Convergence — outstanding orderings by synthesis iteration
# ---------------------------------------------------------------------------

CUs = ["urcu", "qsbr", "wfqueue", "workqueue"]

def _abbrev(x, _):
    if x >= 1_000_000:
        return f"{x/1_000_000:.3g}M"
    if x >= 1_000:
        return f"{x/1_000:.3g}k"
    return f"{x:.3g}"

_abbrev_fmt = mticker.FuncFormatter(_abbrev)

def plot_convergence():
    conv_path = Path(args.csv).parent / "convergence.csv"
    if conv_path.exists():
        conv = pd.read_csv(conv_path)
    else:
        print("No convergence.csv; trying nix log fallback.")
        _, conv = _load_nix_data()
        if conv.empty:
            print("No nix log data found; skipping convergence plot.")
            return
    conv["opt"] = conv["compiler"].str.extract(r"-(O\d)")
    conv["fc"]  = conv["compiler"].str.extract(r"-fc(\d+)").astype(float)
    conv = conv[conv["fc"].notna()].copy()
    if conv.empty:
        return
    conv = conv[conv["cu_name"].isin(CUs)]
    fc_vals  = sorted(conv["fc"].unique())
    max_iters = conv.groupby("cu_name")["iteration"].max()
    cu_names = sorted(conv["cu_name"].unique(), key=lambda cu: max_iters.get(cu, 0))
    palette  = sns.color_palette("tab10", len(fc_vals))
    color_map = dict(zip(fc_vals, palette))

    fig, axes = plt.subplots(
        len(opt_levels), len(cu_names),
        figsize=(3 * len(cu_names), 3 * len(opt_levels)),
        squeeze=False,
    )
    fig.suptitle("Synthesis Convergence (outstanding orderings per iteration)", y=1)

    for row, opt in enumerate(opt_levels):
        for col, cu in enumerate(cu_names):
            ax = axes[row][col]
            sub = conv[(conv["opt"] == opt) & (conv["cu_name"] == cu)]
            if sub.empty:
                ax.set_visible(False)
                continue
            for fc in fc_vals:
                fsub = sub[sub["fc"] == fc]
                if fsub.empty:
                    continue
                mean = fsub.groupby("iteration")["outstanding"].mean().reset_index()
                ax.plot(mean["iteration"], mean["outstanding"],
                        marker=None, label=fc, color=color_map[fc])
            ax.set_title(f"[{opt}] {cu}")
            ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.4)
            ax.yaxis.set_major_formatter(_abbrev_fmt)
            # Annotate dep coverage: mark iteration 0 with a vertical bar and
            # show in the top-right corner what fraction deps already ordered.
            ax.axvline(x=0, color="dimgray", linestyle="--", linewidth=1, zorder=3)
            if "total" in sub.columns:
                iter0 = sub[sub["iteration"] == 0]
                if not iter0.empty:
                    dep_cov = 1.0 - (iter0["outstanding"] / iter0["total"]).mean()
                    ax.text(0.97, 0.96, f"{dep_cov:.0%} ordered\nthrough dep.",
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=11, color="dimgray")

    fig.supxlabel("Synthesis Iteration", y=0.1)
    fig.supylabel("Outstanding Orderings")
    _finalize(fig, output_path(args.output, "convergence"),
              legend_anchor=(0.5, 0.95),
              tight_rect=[0, 0.03, 1, 0.97])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

plot_fc_sweep()
plot_best_vs()
plot_compile_time()
plot_convergence()
