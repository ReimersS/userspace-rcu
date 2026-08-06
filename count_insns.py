#!/usr/bin/env python3
"""Count AArch64 memory-access instructions and fences in compiled libraries."""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# Instruction categories
# ---------------------------------------------------------------------------

CATEGORIES = [
    "relaxed_load", "relaxed_store",
    "acq_load", "acqpc_load", "rel_store",
    "exclusive_load", "exclusive_store",
    "atomic_rmw",
    "fence_full", "fence_ld", "fence_st",
    "dsb",
]

_RELAXED_LOADS = {
    "ldr", "ldrsw", "ldrsb", "ldrsh", "ldur", "ldursw", "ldursb", "ldursh",
    "ldrb", "ldrh", "ldp", "ldpsw",
}
_RELAXED_STORES = {
    "str", "strb", "strh", "stur", "sturb", "sturh", "stp",
}
_ACQ_LOADS = {
    "ldar", "ldarb", "ldarh",
}
_ACQPC_LOADS = {
    "ldapr", "ldaprb", "ldaprh",
}
_REL_STORES = {
    "stlr", "stlrb", "stlrh",
}
_EXCLUSIVE_LOADS = {
    "ldxr", "ldxrb", "ldxrh", "ldaxr", "ldaxrb", "ldaxrh",
}
_EXCLUSIVE_STORES = {
    "stxr", "stxrb", "stxrh", "stlxr", "stlxrb", "stlxrh",
}
_ATOMIC_RMW = set()
for base in ("cas", "swp", "ldadd", "ldclr", "ldset", "ldeor",
             "ldsmax", "ldsmin", "ldumax", "ldumin",
             "stadd", "stclr", "stset", "steor"):
    for order in ("", "a", "l", "al"):
        for size in ("", "b", "h"):
            _ATOMIC_RMW.add(base + order + size)

# Mnemonic → category (for non-barrier instructions)
_MNEMONIC_MAP = {}
for m in _RELAXED_LOADS:
    _MNEMONIC_MAP[m] = "relaxed_load"
for m in _RELAXED_STORES:
    _MNEMONIC_MAP[m] = "relaxed_store"
for m in _ACQ_LOADS:
    _MNEMONIC_MAP[m] = "acq_load"
for m in _ACQPC_LOADS:
    _MNEMONIC_MAP[m] = "acqpc_load"
for m in _REL_STORES:
    _MNEMONIC_MAP[m] = "rel_store"
for m in _EXCLUSIVE_LOADS:
    _MNEMONIC_MAP[m] = "exclusive_load"
for m in _EXCLUSIVE_STORES:
    _MNEMONIC_MAP[m] = "exclusive_store"
for m in _ATOMIC_RMW:
    _MNEMONIC_MAP[m] = "atomic_rmw"

# objdump instruction line: "   addr:  hex_bytes  mnemonic  operands"
_INSN_RE = re.compile(r"^\s+[0-9a-f]+:\s+[0-9a-f]+\s+(\S+)\s*(.*)")


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def classify_line(mnemonic: str, operands: str) -> str | None:
    mn = mnemonic.lower()
    if mn == "dmb":
        op = operands.strip().lower()
        if op == "ishld":
            return "fence_ld"
        if op == "ishst":
            return "fence_st"
        if op in ("ish", "sy", ""):
            return "fence_full"
        return "fence_full"  # unknown variant → full barrier bucket
    if mn == "dsb":
        return "dsb"
    return _MNEMONIC_MAP.get(mn)


def count_binary(path: str) -> dict[str, int]:
    """Run objdump -d on a binary and count instruction categories."""
    r = subprocess.run(
        ["objdump", "-d", path],
        capture_output=True, text=True, timeout=120,
    )
    counts = Counter()
    for line in r.stdout.splitlines():
        m = _INSN_RE.match(line)
        if not m:
            continue
        cat = classify_line(m.group(1), m.group(2))
        if cat:
            counts[cat] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Config & main
# ---------------------------------------------------------------------------

def load_config(path: str):
    with open(path) as f:
        cfg = json.load(f)
    config_dir = os.path.dirname(os.path.abspath(path))
    compilers = []
    for c in cfg["compilers"]:
        result_dir = os.path.join(config_dir, f"result-{c['name']}")
        compilers.append({"name": c["name"], "result_dir": result_dir})
    return compilers


def find_benchmarks(result_dir: str, pattern: str | None) -> list[str]:
    bench_dir = os.path.join(result_dir, "tests", "benchmark")
    if not os.path.isdir(bench_dir):
        return []
    found = []
    for p in sorted(os.listdir(bench_dir)):
        full = os.path.join(bench_dir, p)
        if not os.path.isfile(full) or not os.access(full, os.X_OK):
            continue
        if p.endswith(".sh"):
            continue
        if not p.startswith("test_"):
            continue
        if pattern and not glob.fnmatch.fnmatch(p, pattern):
            continue
        found.append(full)
    return found


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

TABLE_PROGRAMS = [
    ("mutex",    "test_mutex"),
    ("urcu",     "test_urcu"),
    ("mb",       "test_urcu_mb"),
    ("qsbr",     "test_urcu_qsbr"),
]

TABLE_COLS = [
    "relaxed_load", "acqpc_load", "acq_load",
    "relaxed_store", "rel_store",
    "fence_ld", "fence_st", "fence_full",
]

# fc value → size label, ordered L(ow)/H(igh)
FC_ORDER = [(1, "L"), (333, "H")]


def _lookup(rows, program, compiler):
    """Find a row by program binary name and compiler name."""
    for r in rows:
        if r["program"] == program and r["compiler"] == compiler:
            return r
    return None


def _cell(val):
    return str(val)


def _colored_delta(val, plain=False):
    """Color a single delta value. red=more, green=less. plain=no color."""
    if val == 0:
        return "0"
    if plain:
        return f"+{val}" if val > 0 else str(val)
    if val > 0:
        return f"\\cellr{{+{val}}}"
    return f"\\cellg{{{val}}}"


def _orb_cell(vals):
    """Format orb values as L/H."""
    return "/".join(_cell(v) for v in vals)


def _delta_cell(vals, plain=False):
    """Format delta values with color as L/H."""
    return "/".join(_colored_delta(v, plain) for v in vals)

# Columns where more = less sync (inverted color logic)
_INVERT_COLS = {"relaxed_load", "relaxed_store"}


TABLE_ROW_LABELS = [
    ("relaxed_load",  r"$\Rlab$ $\MOrlx$"),
    ("acq_load",      r"$\Rlab$ $\MOacq$"),
    ("relaxed_store", r"$\Wlab$ $\MOrlx$"),
    ("rel_store",     r"$\Wlab$ $\MOrel$"),
    ("fence_ld",      r"$\dmbfull$ $ld$"),
    ("fence_full",    r"$\dmbfull$ full"),
]


def write_latex(rows, out_path, opt="O3"):
    progs = TABLE_PROGRAMS
    ncols = len(TABLE_ROW_LABELS)

    lines = []
    w = lines.append

    # preamble helpers (user should include \usepackage{xcolor,colortbl} in doc)
    w(r"    % requires: \usepackage{xcolor,colortbl}")
    w(r"    \newcommand{\cellg}[1]{\cellcolor{green!15}{#1}}")
    w(r"    \newcommand{\cellr}[1]{\cellcolor{red!15}{#1}}")
    w(r"    \newcommand{\gray}{\cellcolor{gray!10}}")
    w(r"    \setlength{\tabcolsep}{3pt}")
    w("")

    # Total column (sum of all non-relaxed accesses)
    TOTAL_COLS = ["acqpc_load", "acq_load",
                      "rel_store",
                      "exclusive_load", "exclusive_store", "atomic_rmw"]

    short_labels = {
        "relaxed_load": r"$\MOrlx$", "acq_load": r"$\MOacq$",
        "relaxed_store": r"$\MOrlx$", "rel_store": r"$\MOrel$",
        "fence_ld": r"$ld$", "fence_full": r"full",
    }
    groups = [
        (r"$\Rlab$", ["relaxed_load", "acq_load"]),
        (r"$\Wlab$", ["relaxed_store", "rel_store"]),
        (r"$\dmbfull$", ["fence_ld", "fence_full"]),
    ]

    # Each access type: value + Δ = 2 cols; Total: 1 col
    col_spec = "|l|l|c|" + "c c|" * ncols
    w(f"    \\begin{{tabular}}{{{col_spec}}}")
    w(r"        \hline")

    # Header row 1: groups
    w(r"        \multirow{2}{*}{Program} & \multirow{2}{*}{Compiler}"
      r" & \multirow{2}{*}{\shortstack{Total\\non-$\MOrlx$}}"
      + "".join(f" & \\multicolumn{{{2*len(cols)}}}{{c|}}{{{name}}}"
                for name, cols in groups)
      + r"\\")

    # Header row 2: sub-labels per access type
    sub = [""]  # Total — no sub-label
    for _, cols in groups:
        for col_key in cols:
            sub.extend([short_labels[col_key], r"$\Delta$"])
    w(f"        & & {' & '.join(sub)}\\\\")
    w(r"        \hline")
    w(r"        \hline")

    for label, binary in progs:
        clang_r = _lookup(rows, binary, f"clang-{opt}")
        cir_r = _lookup(rows, binary, f"clangir-{opt}")
        orb_rs = [_lookup(rows, binary, f"orb-{opt}-fc{fc}")
                  for fc, _ in FC_ORDER]

        def _total(r):
            return sum(r[c] if r else 0 for c in TOTAL_COLS)

        base_r = cir_r  # clangir is baseline

        # clangir row (baseline, no deltas)
        cir_cells = [_cell(_total(cir_r))]
        for col_key, _ in TABLE_ROW_LABELS:
            val = cir_r[col_key] if cir_r else 0
            cir_cells.extend([_cell(val), ""])
        w(f"        \\multirow{{3}}{{*}}{{{label}}}"
          f" & clangir & {' & '.join(cir_cells)}\\\\[-3pt]")

        # clang row (delta vs clangir)
        clang_cells = [_cell(_total(clang_r))]
        for col_key, _ in TABLE_ROW_LABELS:
            val = clang_r[col_key] if clang_r else 0
            base_val = base_r[col_key] if base_r else 0
            d = val - base_val
            clang_cells.extend([_cell(val),
                              _colored_delta(d, plain=col_key in _INVERT_COLS)])
        w(f"         & clang & {' & '.join(clang_cells)}\\\\[-3pt]")

        # orb row — bold, combined L/H cells, delta vs clangir
        orb_cells = [f"\\gray \\textbf{{{_orb_cell([_total(r) for r in orb_rs])}}}"]
        for col_key, _ in TABLE_ROW_LABELS:
            base_val = base_r[col_key] if base_r else 0
            vals = [r[col_key] if r else 0 for r in orb_rs]
            orb_cells.append(f"\\gray \\textbf{{{_orb_cell(vals)}}}")
            diffs = [v - base_val for v in vals]
            orb_cells.append(_delta_cell(diffs, plain=col_key in _INVERT_COLS))
        w(f"         & \\textbf{{\\system}}"
          f" & {' & '.join(orb_cells)}\\\\")

        w(r"         \hline")

    w(r"    \end{tabular}")
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="JSON config file")
    ap.add_argument("--output", default="insn-counts.csv", help="Output CSV")
    ap.add_argument("--latex", default=None, help="Output LaTeX table file")
    ap.add_argument("--opt", default="O0", help="Optimization level for LaTeX (default: O0)")
    ap.add_argument("--filter", default=None,
                    help="Glob pattern for binaries (default: all test_*)")
    args = ap.parse_args()

    compilers = load_config(args.config)
    rows = []

    for cc in compilers:
        bins = find_benchmarks(cc["result_dir"], args.filter)
        if not bins:
            print(f"warning: no benchmarks found for {cc['name']} "
                  f"at {cc['result_dir']}/tests/benchmark/", file=sys.stderr)
            continue
        for bin_path in bins:
            name = os.path.basename(bin_path)
            counts = count_binary(bin_path)
            row = {"program": name, "compiler": cc["name"]}
            for cat in CATEGORIES:
                row[cat] = counts.get(cat, 0)
            rows.append(row)
            total_mem = sum(row[c] for c in CATEGORIES)
            print(f"  {cc['name']:20s}  {name:35s}  {total_mem:5d} mem insns")

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["program", "compiler"] + CATEGORIES)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows written to {args.output}")

    if args.latex:
        write_latex(rows, args.latex, args.opt)
        print(f"LaTeX table written to {args.latex}")


if __name__ == "__main__":
    main()
