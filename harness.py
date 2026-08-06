#!/usr/bin/env python3
"""Benchmark harness: build urcu with multiple compiler configs, run benchmarks, compare."""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


N_RUNS = 5
N_THREADS = [64]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_compiler(flake_dir: str, compiler: dict) -> tuple[str, str]:
    """nix build the urcu variant and return (store_path, build_log)."""
    flake_ref = f"{flake_dir}#{compiler['flake_output']}.aarch64-linux"
    out_link = Path(flake_dir) / f"result-{compiler['name']}"
    cmd = [
        "nix", "build", flake_ref,
        "--out-link", str(out_link),
        "--print-out-paths",
        "-L",
        "--log-format", "raw",
    ]
    print(f"[build] {compiler['name']}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"nix build failed for {compiler['name']}")
    store_path = result.stdout.strip().splitlines()[-1]
    print(f"[build] {compiler['name']} -> {store_path}")
    return store_path, result.stderr


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_benchmark(store_path: str, benchmark: dict, out_dir: Path,
                  run_n: int, n_threads: int) -> tuple[str, float]:
    """Run a single benchmark .tap script, return (output, elapsed_seconds).

    n_threads limits the CPU affinity mask so that nproc returns n_threads,
    which controls THREAD_MUL inside the tap scripts.
    """
    tap_script = Path(store_path) / benchmark["path"]
    env = os.environ.copy()
    env["URCU_TESTS_SRCDIR"] = str(Path(store_path) / "tests")
    env["URCU_TESTS_BUILDDIR"] = str(Path(store_path) / "tests")
    env["URCU_TESTS_NPROC_BIN"] = "nproc"

    print(f"  [run] {benchmark['name']} t={n_threads} #{run_n}")
    t0 = time.monotonic()
    result = subprocess.run(
        ["taskset", "-c", f"0-{n_threads - 1}", "bash", str(tap_script)],
        capture_output=True, text=True, env=env,
    )
    elapsed = time.monotonic() - t0

    output = result.stdout + result.stderr
    out_file = out_dir / f"{benchmark['name']}.{n_threads}t.{run_n}.tap"
    out_file.write_text(output)
    if result.returncode != 0:
        print(f"  [warn] {benchmark['name']} t={n_threads} #{run_n} exited {result.returncode}")
    return output, elapsed


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

_SUMMARY_RE = re.compile(
    r"^# SUMMARY\s+(?P<binary>\S+)\s+"
    r"testdur\s+(?P<testdur>\d+)\s+"
    r"nr_readers\s+(?P<nr_readers>\d+)\s+"
    r"rdur\s+\d+\s+"
    r"(?:wdur\s+\d+\s+)?"                          # urcu tests only
    r"nr_writers\s+(?P<nr_writers>\d+)\s+"
    r"wdelay\s+\d+\s+"
    r"nr_reads\s+(?P<nr_reads>\d+)\s+"
    r"nr_writes\s+(?P<nr_writes>\d+)\s+"
    r"nr_ops\s+(?P<nr_ops>\d+)"
    r"(?:\s+batch\s+(?P<batch>\d+))?"              # urcu tests only
    r"(?:.*?nr_add\s+(?P<nr_add>\d+))?"            # hashtable tests only
    r"(?:.*?nr_add_fail\s+(?P<nr_add_fail>\d+))?"  # hashtable tests only
    r"(?:.*?nr_remove\s+(?P<nr_remove>\d+))?"      # hashtable tests only
    r"(?:.*?nr_leaked\s+(?P<nr_leaked>-?\d+))?",   # hashtable tests only
)

_TAP_LINE_RE = re.compile(r"^(ok|not ok)\s+(\d+)\s+-\s+(?:time\s+)?(.+)$")
_TAP_OK_RE = re.compile(r"^(ok|not ok)\b", re.MULTILINE)


def _hash_flags(cmd: str) -> tuple[str, str, str]:
    """Extract write_mode, mm_backend, and table size category from a test_urcu_hash command line.

    Returns a short label: "{write_mode}-{size}-{backend}[-suffix...]"
    Suffixes: "wo" (write-only, 0 readers), "rz" (resize variant with -k), "C<N>" (-C flag).
    """
    args = cmd.split()

    def _arg(flag):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                return args[idx + 1]
        return None

    if "-s" in args:
        write_mode = "replace"
    elif "-u" in args:
        write_mode = "add_uniq"
    elif "-i" in args:
        write_mode = "add_only"
    elif "-U" in args:
        write_mode = "urcu_only"
    else:
        write_mode = "add_del"

    mm_backend = _arg("-B") or "default"

    m_val = _arg("-M")
    if m_val is not None:
        m = int(m_val)
        size = "S" if m <= 1 else ("M" if m <= 10 else "L")
    elif "-R" in args:
        size = "resize"
    else:
        size = "def"

    suffixes = []
    # binary is args[0], positional args[1]=nr_readers
    if len(args) > 1 and args[1] == "0":
        suffixes.append("wo")
    if "-k" in args:
        suffixes.append("rz")
    c_val = _arg("-C")
    if c_val is not None:
        suffixes.append(f"C{c_val}")

    label = f"{write_mode}-{size}-{mm_backend}"
    if suffixes:
        label += "-" + "-".join(suffixes)
    return label, mm_backend, size


def _urcu_flags(cmd: str) -> tuple[int | None, int | None]:
    """Extract writer_delay (-d) and reader_cs_dur (-c) from a urcu test command line."""
    args = cmd.split()
    def _get(flag):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                try:
                    return int(args[idx + 1])
                except ValueError:
                    pass
        return None
    return _get("-d"), _get("-c")


def parse_summaries(output: str) -> list[dict]:
    rows = []
    lines = output.splitlines()
    for i, line in enumerate(lines):
        m = _SUMMARY_RE.match(line)
        if not m:
            continue
        tap_n, tap_ok, tap_cmd = None, None, None
        for j in range(i + 1, min(i + 3, len(lines))):
            t = _TAP_LINE_RE.match(lines[j])
            if t:
                tap_ok  = t.group(1) == "ok"
                tap_n   = int(t.group(2))
                tap_cmd = t.group(3)
                break
        def opt_int(key):
            v = m.group(key)
            return int(v) if v is not None else None
        prog = Path(m.group("binary")).name.removeprefix("test_")
        if prog == "urcu_hash" and tap_cmd:
            prog, mm_backend, size = _hash_flags(tap_cmd)
            write_mode = prog.split("-")[0]
            writer_delay = reader_cs_dur = None
        else:
            write_mode = mm_backend = size = None
            writer_delay, reader_cs_dur = _urcu_flags(tap_cmd) if tap_cmd else (None, None)
        testdur = int(m.group("testdur"))
        nr_ops  = int(m.group("nr_ops"))
        rows.append({
            "tap_n":          tap_n,
            "tap_ok":         tap_ok,
            "test_program":   prog,
            "write_mode":     write_mode,
            "mm_backend":     mm_backend,
            "writer_delay":   writer_delay,
            "reader_cs_dur":  reader_cs_dur,
            "testdur":        testdur,
            "nr_readers":     int(m.group("nr_readers")),
            "nr_writers":     int(m.group("nr_writers")),
            "nr_reads":       int(m.group("nr_reads")),
            "nr_writes":      int(m.group("nr_writes")),
            "nr_ops":         nr_ops,
            "throughput":     nr_ops / testdur if testdur else 0,
            "batch":          opt_int("batch"),
            "nr_add":         opt_int("nr_add"),
            "nr_remove":      opt_int("nr_remove"),
            "nr_leaked":      opt_int("nr_leaked"),
        })
    return rows


def tap_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {"ok": 0, "not ok": 0}
    for m in _TAP_OK_RE.finditer(output):
        counts[m.group(1)] += 1
    return counts


# ---------------------------------------------------------------------------
# Synthesis log parsing
# ---------------------------------------------------------------------------

_FS_REQUIRED = re.compile(r'\[FenceSynthesis\].*?required pairs=(\d+)')
_FS_MATRIX   = re.compile(r'\[FenceSynthesis\].*?n=(\d+)\s+unreachable=(\d+)\s+reachable=(\d+)')
_FS_ORDERED  = re.compile(r'\[FenceSynthesis\].*?ordered=(\d+)/\d+ overspecified=(\d+).*?(?:batched=(\d+))?.*?t=(\d+)ms')
_FS_DONE     = re.compile(r'\[FenceSynthesis\].*?done ordered=(\d+)/\d+ overspecified=(\d+).*?t=(\d+)ms')
# Autotools CC step: "  CC       liburcu_la-urcu.lo" → extract short name "urcu"
_CC_STEP     = re.compile(r'\bCC\b\s+\S*?(?:la-)?(\w+)\.lo\b')


def parse_synthesis_log(build_log: str) -> tuple[list[dict], list[dict]]:
    """Parse [FenceSynthesis] lines from a nix build log.

    Returns (synth_rows, conv_rows):
      synth_rows  — one dict per compilation unit (as before, with added cu_name)
      conv_rows   — one dict per (CU, iteration): cu_name, iteration, outstanding
    """
    synth_rows: list[dict] = []
    conv_rows:  list[dict] = []
    unit: dict | None = None
    current_cu: str = ""
    iteration: int = 0

    for raw_line in build_log.splitlines():
        # Nix prefixes build log lines with "<pkg>> "; strip it.
        line = re.sub(r'^[^>]+> ?', '', raw_line)

        # Track which .lo is being compiled so we can label the CU.
        m = _CC_STEP.search(line)
        if m and '[FenceSynthesis]' not in line:
            current_cu = m.group(1)
            continue

        m = _FS_REQUIRED.search(line)
        if m:
            iteration = 0
            unit = {
                "cu_name": current_cu or "unknown",
                "source": int(m.group(1)),
                "reachable": None,
                "implicit": None,
                "overspecified_initial": None,
                "ordered_final": None,
                "overspecified_final": None,
                "promotions": -1,  # first ordered= line is implicit (not a promotion)
                "synthesis_time_ms": None,
            }
            continue

        if unit is None:
            continue

        m = _FS_MATRIX.search(line)
        if m:
            unit["reachable"] = int(m.group(3))
            continue

        m = _FS_DONE.search(line)
        if m:
            unit["ordered_final"] = int(m.group(1))
            unit["overspecified_final"] = int(m.group(2))
            unit["synthesis_time_ms"] = int(m.group(3))
            synth_rows.append(unit)
            unit = None
            continue

        m = _FS_ORDERED.search(line)
        if m:
            ordered = int(m.group(1))
            batched = int(m.group(3)) if m.group(3) else 1
            if unit["implicit"] is None:
                unit["implicit"] = ordered
                unit["overspecified_initial"] = int(m.group(2))
            unit["promotions"] += batched
            outstanding = unit["source"] - ordered
            conv_rows.append({
                "cu_name": unit["cu_name"],
                "iteration": iteration,
                "outstanding": outstanding,
                "overspecified": int(m.group(2)),
                "reachable": unit.get("reachable"),
                "batched": batched,
            })
            iteration += 1

    return synth_rows, conv_rows


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def compare_outputs(all_results: dict) -> str:
    """Compare TAP pass/fail counts against first compiler baseline."""
    lines = []
    baseline_name = next(iter(all_results))
    lines.append(f"Baseline compiler: {baseline_name}\n")

    for benchmark_name in next(iter(all_results.values())):
        lines.append(f"\n=== {benchmark_name} ===")
        combined_baseline = "".join(
            o for o, *_ in all_results[baseline_name][benchmark_name]["outputs"]
        )
        baseline_counts = tap_counts(combined_baseline)
        lines.append(
            f"  {baseline_name:14s}: ok={baseline_counts['ok']}  "
            f"not_ok={baseline_counts['not ok']}"
        )
        for compiler, benchmarks in all_results.items():
            if compiler == baseline_name:
                continue
            combined = "".join(o for o, *_ in benchmarks[benchmark_name]["outputs"])
            counts = tap_counts(combined)
            match = (
                counts["ok"] == baseline_counts["ok"]
                and counts["not ok"] == baseline_counts["not ok"]
            )
            status = "OK" if match else "MISMATCH"
            lines.append(
                f"  {compiler:14s}: ok={counts['ok']}  "
                f"not_ok={counts['not ok']}  [{status}]"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "run_id", "compiler", "n_threads", "run_n", "benchmark",
    "tap_n", "tap_ok", "test_program", "write_mode", "mm_backend",
    "writer_delay", "reader_cs_dur",
    "nr_readers", "nr_writers",
    "nr_ops", "nr_reads", "nr_writes",
    "testdur", "throughput_ops_per_sec", "wall_clock_s",
    "batch", "nr_add", "nr_remove", "nr_leaked",
]


def write_csv(run_id: str, all_results: dict, path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for compiler, benchmarks in all_results.items():
            for benchmark_name, data in benchmarks.items():
                for run_n, (output, wall, n_threads) in enumerate(data["outputs"], 1):
                    wall_str = f"{wall:.2f}" if wall is not None else ""
                    for summary in parse_summaries(output):
                        writer.writerow({
                            "run_id":                 run_id,
                            "compiler":               compiler,
                            "n_threads":              n_threads,
                            "run_n":                  run_n,
                            "benchmark":              benchmark_name,
                            "tap_n":                  summary["tap_n"],
                            "tap_ok":                 summary["tap_ok"],
                            "test_program":           summary["test_program"],
                            "write_mode":             summary["write_mode"],
                            "mm_backend":             summary["mm_backend"],
                            "writer_delay":           summary["writer_delay"],
                            "reader_cs_dur":          summary["reader_cs_dur"],
                            "nr_readers":             summary["nr_readers"],
                            "nr_writers":             summary["nr_writers"],
                            "nr_ops":                 summary["nr_ops"],
                            "nr_reads":               summary["nr_reads"],
                            "nr_writes":              summary["nr_writes"],
                            "testdur":                summary["testdur"],
                            "throughput_ops_per_sec": summary["throughput"],
                            "wall_clock_s":           wall_str,
                            "batch":                  summary["batch"],
                            "nr_add":                 summary["nr_add"],
                            "nr_remove":              summary["nr_remove"],
                            "nr_leaked":              summary["nr_leaked"],
                        })


SYNTH_FIELDS = [
    "run_id", "compiler", "cu_name",
    "source", "reachable", "implicit", "overspecified_initial",
    "ordered_final", "overspecified_final",
    "promotions", "synthesis_time_ms",
]

CONV_FIELDS = ["run_id", "compiler", "cu_name", "iteration", "outstanding",
               "overspecified", "reachable"]


def write_synthesis_csv(run_id: str, rows: list[dict], path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SYNTH_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SYNTH_FIELDS})
    print(f"Synthesis: {path} ({len(rows)} rows)")


def write_convergence_csv(run_id: str, rows: list[dict], path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CONV_FIELDS})
    print(f"Convergence: {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Symlinks
# ---------------------------------------------------------------------------

def update_symlinks(runs_dir: Path, run_dir: Path) -> None:
    """Keep 'latest' and 'previous' symlinks up to date."""
    latest = runs_dir / "latest"
    previous = runs_dir / "previous"

    # Rotate: old latest becomes previous
    if latest.is_symlink():
        target = latest.resolve()
        _replace_symlink(previous, target.name)

    _replace_symlink(latest, run_dir.name)


def _replace_symlink(link: Path, target_name: str) -> None:
    tmp = link.with_suffix(".tmp")
    tmp.symlink_to(target_name)
    tmp.replace(link)


# ---------------------------------------------------------------------------
# Re-parse
# ---------------------------------------------------------------------------

def reparse(run_dir: Path) -> None:
    """Re-parse .tap files in an existing run directory and rewrite results."""
    run_id = run_dir.name
    all_results: dict[str, dict] = {}

    for compiler_dir in sorted(run_dir.iterdir()):
        if not compiler_dir.is_dir():
            continue
        compiler = compiler_dir.name
        all_results[compiler] = {}

        # Group *.{n_threads}t.{run_n}.tap (or legacy *.{run_n}.tap) by benchmark.
        tap_groups: dict[str, list[Path]] = defaultdict(list)
        for tap_file in sorted(compiler_dir.glob("*.tap")):
            bench = tap_file.name.split(".")[0]
            tap_groups[bench].append(tap_file)

        _threads_re = re.compile(r"\.(\d+)t\.\d+\.tap$")
        for bench, files in sorted(tap_groups.items()):
            outputs = []
            for f in sorted(files):
                m = _threads_re.search(f.name)
                n_threads = int(m.group(1)) if m else None
                outputs.append((f.read_text(), None, n_threads))
            all_results[compiler][bench] = {"outputs": outputs}

    report = compare_outputs(all_results)
    print(report)
    (run_dir / "report.txt").write_text(report)

    csv_path = run_dir / "results.csv"
    write_csv(run_id, all_results, csv_path)
    print(f"\nResults: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="urcu benchmark harness")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--reparse", metavar="RUN_DIR",
                        help="re-parse .tap files in an existing run directory")
    args = parser.parse_args()

    if args.reparse:
        reparse(Path(args.reparse))
        return

    config = json.loads(Path(args.config).read_text())
    flake_dir = config["flake_dir"]
    compilers = config["compilers"]
    benchmarks = config["benchmarks"]

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runs_dir = Path(args.runs_dir)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    print(f"Run directory: {run_dir}")

    # Build all compiler variants first
    store_paths: dict[str, str] = {}
    build_logs:  dict[str, str] = {}
    for compiler in compilers:
        path, log = build_compiler(flake_dir, compiler)
        store_paths[compiler["name"]] = path
        build_logs[compiler["name"]]  = log

    # Run benchmarks (N_RUNS times each)
    all_results: dict[str, dict] = {}
    for compiler in compilers:
        name = compiler["name"]
        store_path = store_paths[name]
        compiler_dir = run_dir / name
        compiler_dir.mkdir()
        all_results[name] = {}
        print(f"\n[{name}]")
        for benchmark in benchmarks:
            outputs = []
            for n_threads in N_THREADS:
                for run_n in range(1, N_RUNS + 1):
                    output, elapsed = run_benchmark(store_path, benchmark,
                                                    compiler_dir, run_n, n_threads)
                    outputs.append((output, elapsed, n_threads))
            all_results[name][benchmark["name"]] = {"outputs": outputs}

    # Compare and report
    report = compare_outputs(all_results)
    print(f"\n{report}")
    (run_dir / "report.txt").write_text(report)

    # Results CSV
    csv_path = run_dir / "results.csv"
    write_csv(run_id, all_results, csv_path)
    print(f"\nResults: {csv_path}")

    # Synthesis + convergence CSVs (orb compilers only, skipped on nix cache hits)
    synth_rows: list[dict] = []
    conv_rows:  list[dict] = []
    for compiler in compilers:
        if not compiler.get("synthesis"):
            continue
        log = build_logs[compiler["name"]]
        if not log:
            continue
        s_rows, c_rows = parse_synthesis_log(log)
        for r in s_rows:
            r["compiler"] = compiler["name"]
            r["run_id"]   = run_id
        for r in c_rows:
            r["compiler"] = compiler["name"]
            r["run_id"]   = run_id
        synth_rows.extend(s_rows)
        conv_rows.extend(c_rows)
    if synth_rows:
        write_synthesis_csv(run_id, synth_rows, run_dir / "synthesis.csv")
    if conv_rows:
        write_convergence_csv(run_id, conv_rows, run_dir / "convergence.csv")

    update_symlinks(runs_dir, run_dir)


if __name__ == "__main__":
    main()
