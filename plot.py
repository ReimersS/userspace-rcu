#!/usr/bin/env python3
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Programs to show in the reduced view (edit as needed)
REDUCED = {
    "hashtable_3s": [
        "replace-M-default-rz",
        "add_del-S-default",
        "add_del-M-default",
        "add_uniq-def-default",
        "add_del-def-default",
        "add_del-resize-default",
    ],
    "urcu_3s": [
        "urcu_qsbr",
        "urcu_qsbr_gc",
        "urcu",
        "urcu_gc",
        "urcu_mb",
        "perthreadlock",
    ],
}

parser = argparse.ArgumentParser()
parser.add_argument("csv", nargs="?", default="runs/latest/results.csv")
parser.add_argument("--output", default=None,
                    help="output file base; -full and -reduced suffixes are appended")
args = parser.parse_args()

df = pd.read_csv(args.csv)


benchmarks = list(df["benchmark"].unique())
opt_levels = ["O0", "O3"]
df["opt"] = df["compiler"].str.extract(r"-(O\d+)$")


def make_plot(data, program_filter=None, output=None):
    fig, axes = plt.subplots(
        len(opt_levels), len(benchmarks),
        figsize=(7 * len(benchmarks), 5 * len(opt_levels)),
    )

    handles, labels = None, None

    for row, opt in enumerate(opt_levels):
        for col, bname in enumerate(benchmarks):
            ax = axes[row][col]
            sub = data[(data["benchmark"] == bname) & (data["opt"] == opt)]
            if program_filter:
                keep = program_filter.get(bname, [])
                sub = sub[sub["test_program"].isin(keep)]
            order = sorted(sub["test_program"].unique())
            sns.barplot(
                data=sub,
                y="test_program",
                x="throughput_ops_per_sec",
                hue="compiler",
                palette="pastel",
                order=order,
                ax=ax,
            )
            ax.set_title(f"{bname} [{opt}]")
            ax.set_ylabel("")
            ax.set_xlabel("throughput (ops/s)")
            ax.tick_params(axis="y", labelsize=7)
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

    fig.legend(handles, labels, loc="upper center", ncols=len(labels),
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.legend(handles, labels, loc="lower center", ncols=len(labels),
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if output:
        plt.savefig(output)
    else:
        plt.show()
    plt.close(fig)


def output_path(template, suffix):
    if template is None:
        return None
    base, _, ext = template.rpartition(".")
    return f"{base}-{suffix}.{ext}" if base else f"{template}-{suffix}"


make_plot(df, output=output_path(args.output, "full"))
make_plot(df, program_filter=REDUCED, output=output_path(args.output, "reduced"))
