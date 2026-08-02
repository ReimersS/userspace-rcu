#!/usr/bin/env python3
"""
litmus_diff.py - Compile C litmus tests through multiple compilers,
extract AArch64 memory skeletons, emit herd7 .litmus files, compare results.

Usage:
    python3 litmus_diff.py --config config.json
    python3 litmus_diff.py --config config.json --batch <dir>
    python3 litmus_diff.py <file.litmus> --clang /path/to/clang
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Callee-saved registers for pinning local variables.
PIN_REGS = [f"w{i}" for i in range(19, 29)]

# Instruction classification for the skeleton extractor.
LOAD_MNEMONICS = {
    "ldr", "ldrsw", "ldar", "ldarb", "ldarh", "ldapr", "ldaprb", "ldaprh",
    "ldur", "ldxr", "ldxrb", "ldxrh", "ldaxr", "ldaxrb", "ldaxrh",
}
STORE_MNEMONICS = {
    "str", "strb", "strh", "stlr", "stlrb", "stlrh", "stur",
    "stxr", "stxrb", "stxrh", "stlxr", "stlxrb", "stlxrh",
}
CAS_MNEMONICS = {
    "cas", "casa", "casl", "casal",
    "casb", "casab", "caslb", "casalb",
    "cash", "casah", "caslh", "casalh",
}
EMIT_MNEMONICS = {
    "dmb", "dsb", "cbz", "cbnz", "cmp",
    "b.eq", "b.ne", "b.cc", "b.cs",
    "csel", "cinc", "clrex",
}
IMM_MOV_RE = re.compile(r"w\d+,\s*#", re.IGNORECASE)
REG_MOV_RE = re.compile(r"(x\d+),\s*(x\d+)$", re.IGNORECASE)
BRANCH_RE = re.compile(r"^b(\.\w+)?$")
MEM_RE = re.compile(
    r"\[(\w+)(?:,\s*#(-?\d+))?\]!?|"
    r"\[(\w+)\],\s*#(-?\d+)")
FP_SETUP_RE = re.compile(r"add\s+x29,\s*sp,\s*#(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Thread:
    proc_id: int
    params: list          # [(type_str, name), ...]
    body: str
    local_vars: list = field(default_factory=list)

@dataclass
class CLitmus:
    name: str
    init: dict
    threads: list
    condition: str

@dataclass
class AsmInstr:
    mnemonic: str
    operands: str
    label: str = ""

@dataclass
class CompilerConfig:
    name: str
    clang: str
    cflags: list


# ---------------------------------------------------------------------------
# Parse C litmus
# ---------------------------------------------------------------------------

def parse_c_litmus(path: str) -> CLitmus:
    text = Path(path).read_text()

    m = re.match(r"^C\s+(\S+)", text)
    if not m:
        raise ValueError(f"Not a C litmus test: {path}")
    name = m.group(1)

    init = {}
    m = re.search(r"\{([^}]*)\}", text)
    if m:
        for var, val in re.findall(r"\[(\w+)\]\s*=\s*(\d+)", m.group(1)):
            init[var] = int(val)

    threads = []
    for m in re.finditer(r"P(\d+)\s*\(([^)]*)\)\s*\{", text):
        proc_id = int(m.group(1))
        params = _parse_params(m.group(2))
        start, depth, i = m.end(), 1, m.end()
        while i < len(text) and depth > 0:
            if text[i] == "{": depth += 1
            elif text[i] == "}": depth -= 1
            i += 1
        body = text[start:i - 1].strip()
        local_vars = re.findall(r"int\s+(\w+)\s*=", body)
        threads.append(Thread(proc_id, params, body, local_vars))

    m = re.search(r"(exists\s*\(.*\))", text, re.DOTALL)
    condition = m.group(1).strip() if m else "exists (true)"
    return CLitmus(name, init, threads, condition)


def _parse_params(param_str: str) -> list:
    params = []
    for p in param_str.split(","):
        p = p.strip()
        if not p:
            continue
        m = re.match(r"(.+?)\s*\*\s*(\w+)$", p)
        if m:
            params.append((m.group(1).replace(" ", "") + "*", m.group(2)))
        else:
            parts = p.rsplit(None, 1)
            params.append((parts[0].replace(" ", ""), parts[1]) if len(parts) == 2
                          else ("int*", parts[0]))
    return params


# ---------------------------------------------------------------------------
# Wrap as compilable C
# ---------------------------------------------------------------------------

def _rewrite_non_explicit_atomics(body: str) -> str:
    body = re.sub(r"atomic_load\(([^)]+)\)",
                  r"atomic_load_explicit(\1, memory_order_seq_cst)", body)
    body = re.sub(r"atomic_store\(([^)]+)\)",
                  r"atomic_store_explicit(\1, memory_order_seq_cst)", body)
    return body


def _params_used_in_atomics(body: str) -> set:
    return {m.group(1) for m in re.finditer(
        r"atomic_(?:compare_exchange|exchange|fetch_\w+|load|store)\w*\((\w+)", body)}


def _c_type_for_param(typ: str, name: str, atomic_params: set) -> str:
    if name in atomic_params and "atomic_int" not in typ:
        return "atomic_int *"
    if "atomic_int" in typ:
        return "atomic_int *"
    if "volatile" in typ:
        return "volatile int *"
    return "int *"


def wrap_as_c(litmus: CLitmus) -> str:
    lines = ["#include <stdatomic.h>", ""]
    for t in litmus.threads:
        atomic_params = _params_used_in_atomics(t.body)
        c_params = [f"{_c_type_for_param(typ, name, atomic_params)}{name}"
                    for typ, name in t.params]
        body = _rewrite_non_explicit_atomics(t.body)

        reg_decls = []
        for i, rv in enumerate(t.local_vars[:len(PIN_REGS)]):
            reg_decls.append(f'  register int {rv} asm("{PIN_REGS[i]}") = 0;')
            body = re.sub(rf"\bint\s+({re.escape(rv)}\s*=)", r"\1", body)

        keep_alive = ""
        if t.local_vars:
            constraints = ", ".join(f'"r"({rv})' for rv in t.local_vars[:len(PIN_REGS)])
            keep_alive = f'  asm volatile("" :: {constraints});'

        lines.append(f'void P{t.proc_id}({", ".join(c_params)}) {{')
        lines.extend(reg_decls)
        lines.append(f"  {body}")
        if keep_alive:
            lines.append(keep_alive)
        lines.append("}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    default_clang = cfg.get("orb_cc", "clang")
    if not os.path.isabs(default_clang):
        default_clang = os.path.join(config_dir, default_clang)

    compilers = []
    for entry in cfg.get("compilers", []):
        clang = entry.get("clang", default_clang)
        if not os.path.isabs(clang):
            clang = os.path.join(config_dir, clang)
        cflags = entry.get("cflags", [])
        if isinstance(cflags, str):
            cflags = cflags.split()
        compilers.append(CompilerConfig(entry["name"], clang, cflags))

    return {
        "compilers": compilers,
        "herd7": cfg.get("herd7", ""),
        "cat": cfg.get("cat", ""),
        "libdir": cfg.get("libdir", ""),
        "litmus_tests": cfg.get("litmus_tests", ""),
    }


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

def compile_to_asm(c_source: str, compiler: CompilerConfig) -> str:
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(c_source)
        c_path = f.name
    try:
        cmd = [compiler.clang, "-target", "aarch64-linux-gnu",
               "-march=armv8.1-a", "-mno-outline-atomics",
               "-Wno-incompatible-pointer-types", "-S", "-o", "-",
               *compiler.cflags, c_path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0 and not r.stdout.strip():
            raise RuntimeError(r.stderr)
        return r.stdout
    finally:
        os.unlink(c_path)


# ---------------------------------------------------------------------------
# Extract memory skeleton (linear pass with register tracking)
# ---------------------------------------------------------------------------

def _parse_asm_line(line: str):
    """Return (mnemonic, operands) or None."""
    if not line or line.startswith(".") or line.startswith("//"):
        return None
    parts = line.split(None, 1)
    if not parts:
        return None
    mn = parts[0].lower().rstrip(":")
    ops = re.sub(r"//.*", "", parts[1]).strip() if len(parts) > 1 else ""
    return mn, ops


def _parse_mem(ops: str):
    """Extract (base_reg, offset) from memory operand, or None."""
    m = MEM_RE.search(ops)
    if not m:
        return None
    if m.group(1) is not None:
        return m.group(1).lower(), int(m.group(2) or 0)
    return m.group(3).lower(), int(m.group(4) or 0)


def _collect_raw_lines(asm_text: str) -> dict:
    """Split assembly into per-function raw line lists."""
    funcs = {}
    cur, lines = None, []
    for raw in asm_text.splitlines():
        line = raw.strip()
        m = re.match(r"^(P\d+):", line)
        if m:
            if cur:
                funcs[cur] = lines
            cur, lines = m.group(1), []
            continue
        if line.startswith(".Lfunc_end") or line.startswith(".size"):
            if cur:
                funcs[cur] = lines
            cur, lines = None, []
            continue
        if cur is not None:
            lines.append(line)
    if cur:
        funcs[cur] = lines
    return funcs


def _scan_stack_layout(raw_lines: list) -> tuple:
    """Collect fp_offset and 32-bit stack slot offsets (sp-relative).
    64-bit pointer spills are tracked via reg_map, not emitted."""
    fp_offset = 0
    offsets = set()
    for line in raw_lines:
        m = FP_SETUP_RE.match(line.strip())
        if m:
            fp_offset = int(m.group(1))
            continue
        parsed = _parse_asm_line(line.strip())
        if not parsed:
            continue
        mn, ops = parsed
        if mn in ("stp", "ldp"):
            continue
        if mn not in LOAD_MNEMONICS and mn not in STORE_MNEMONICS:
            continue
        mem = _parse_mem(ops)
        if not mem:
            continue
        base, off = mem
        val_reg = ops.split(",")[0].strip().lower()
        if val_reg.startswith("x"):
            continue
        if base == "sp":
            offsets.add(off)
        elif base == "x29":
            offsets.add(off + fp_offset)
    return fp_offset, sorted(offsets)


def _remap_base(ops: str, base: str, mapped: str) -> str:
    """Replace [base] with [mapped] in operand string."""
    return re.sub(r"\[" + re.escape(base) + r"\]",
                  f"[{mapped}]", ops, flags=re.IGNORECASE)


def _emit_stack_access(mn, val_reg, byte_off, stack_base, scratch, is_load):
    """Emit ADD+LDR/STR pair for a stack array access."""
    mn = mn.replace("ldur", "ldr").replace("stur", "str")
    if byte_off == 0:
        return [AsmInstr(mn, f"{val_reg}, [{stack_base}]")]
    return [AsmInstr("add", f"{scratch}, {stack_base}, #{byte_off}"),
            AsmInstr(mn, f"{val_reg}, [{scratch}]")]


def _extract_one_function(raw_lines: list, num_params: int) -> tuple:
    """Linear pass: build skeleton with register tracking.
    Returns (instrs, stack_offsets)."""
    fp_offset, stack_offsets = _scan_stack_layout(raw_lines)
    offset_to_idx = {off: i for i, off in enumerate(stack_offsets)}

    arg_regs = {f"x{i}" for i in range(num_params)}
    reg_map = {f"x{i}": f"x{i}" for i in range(num_params)}
    slot_ptrs = {}

    stack_base = f"x{num_params}"
    scratch = f"x{num_params + 1}"
    instrs = []

    for line in raw_lines:
        line = line.strip()

        # Labels
        m = re.match(r"^(\.L\w+):", line)
        if m:
            instrs.append(AsmInstr("", "", label=m.group(1)))
            continue

        parsed = _parse_asm_line(line)
        if not parsed:
            continue
        mn, ops = parsed

        # Skip frame management
        if mn in ("sub", "add") and "sp" in ops.lower():
            continue
        if mn in ("stp", "ldp", "ret"):
            continue

        # Register moves
        if mn == "mov":
            m_mov = REG_MOV_RE.match(ops)
            if m_mov:
                dst, src = m_mov.group(1).lower(), m_mov.group(2).lower()
                if src in reg_map:
                    reg_map[dst] = reg_map[src]
                continue
            if IMM_MOV_RE.match(ops):
                instrs.append(AsmInstr(mn, ops))
            continue

        if mn in ("movz", "movn"):
            instrs.append(AsmInstr(mn, ops))
            continue

        # Loads and stores
        if mn in LOAD_MNEMONICS or mn in STORE_MNEMONICS:
            mem = _parse_mem(ops)
            if not mem:
                instrs.append(AsmInstr(mn, ops))
                continue

            base, off = mem
            val_reg = ops.split(",")[0].strip().lower()
            is_load = mn in LOAD_MNEMONICS

            if base in ("sp", "x29"):
                sp_off = (off + fp_offset) if base == "x29" else off
                xval = re.sub(r"^w", "x", val_reg)

                # 64-bit: pointer spill/reload → reg_map only
                if val_reg.startswith("x"):
                    if is_load and sp_off in slot_ptrs:
                        reg_map[xval] = slot_ptrs[sp_off]
                    elif not is_load and xval in reg_map and reg_map[xval] in arg_regs:
                        slot_ptrs[sp_off] = reg_map[xval]
                    continue

                # 32-bit: value spill/reload → emit as array access
                idx = offset_to_idx.get(sp_off)
                if idx is not None:
                    instrs.extend(_emit_stack_access(
                        mn, val_reg, idx * 4, stack_base, scratch, is_load))
            else:
                # Real memory access → remap base to arg register
                mapped = reg_map.get(base, base)
                instrs.append(AsmInstr(mn, _remap_base(ops, base, mapped)))
            continue

        # CAS → remap base register
        if mn in CAS_MNEMONICS:
            mem = _parse_mem(ops)
            if mem:
                base, _ = mem
                mapped = reg_map.get(base, base)
                instrs.append(AsmInstr(mn, _remap_base(ops, base, mapped)))
            else:
                instrs.append(AsmInstr(mn, ops))
            continue

        # Barriers, branches, comparisons
        if mn in EMIT_MNEMONICS or BRANCH_RE.match(mn):
            instrs.append(AsmInstr(mn, ops))

    return instrs, stack_offsets


def _pin_value_registers(instrs: list, thread: Thread) -> list:
    """Rename value registers in skeleton to match pinned convention (w19, w20, ...).

    ClangIR ignores asm register constraints, so loads may end up in scratch
    registers like w8 instead of the pinned w19.  This pass detects such
    registers and renames them so the exists condition (which references the
    pinned names) matches the skeleton.
    """
    if not thread.local_vars:
        return instrs

    num_params = len(thread.params)
    param_regs = {f"w{i}" for i in range(num_params)} | {f"x{i}" for i in range(num_params)}
    special = {"wzr", "xzr", "sp", "x29"}
    # Stack base and scratch registers used by the stack array encoding
    stack_regs = {f"w{num_params}", f"x{num_params}",
                  f"w{num_params+1}", f"x{num_params+1}"}
    ignore = param_regs | special | stack_regs
    pinned = {f"w{19+i}" for i in range(len(thread.local_vars))}

    # Collect unique value registers (load destinations) in order of appearance
    seen = []
    seen_set = set()
    for ins in instrs:
        if ins.label or not ins.operands:
            continue
        mn = ins.mnemonic.lower()
        if mn in LOAD_MNEMONICS:
            dst = ins.operands.split(",")[0].strip().lower()
            wreg = re.sub(r"^x", "w", dst)
            if wreg not in ignore and wreg not in pinned and wreg not in seen_set:
                seen.append(wreg)
                seen_set.add(wreg)
        elif mn == "mov" and IMM_MOV_RE.match(ins.operands):
            dst = ins.operands.split(",")[0].strip().lower()
            wreg = re.sub(r"^x", "w", dst)
            if wreg not in ignore and wreg not in pinned and wreg not in seen_set:
                seen.append(wreg)
                seen_set.add(wreg)

    if not seen:
        return instrs

    # Build rename map: actual_reg → pinned_reg
    rename = {}
    for i, wreg in enumerate(seen):
        if i >= len(thread.local_vars):
            break
        pin_w = f"w{19+i}"
        pin_x = f"x{19+i}"
        rename[wreg] = pin_w
        rename[wreg.replace("w", "x", 1)] = pin_x

    if not rename:
        return instrs

    # Apply rename to all instructions
    def _rename_operands(ops: str) -> str:
        for old, new in sorted(rename.items(), key=lambda x: -len(x[0])):
            ops = re.sub(rf'\b{re.escape(old)}\b', new, ops, flags=re.IGNORECASE)
        return ops

    return [AsmInstr(ins.mnemonic, _rename_operands(ins.operands), ins.label)
            if not ins.label else ins
            for ins in instrs]


def extract_functions(asm_text: str, litmus: CLitmus) -> dict:
    """Extract per-function skeletons. Returns {name: (instrs, stack_offsets)}."""
    raw = _collect_raw_lines(asm_text)
    result = {}
    for t in litmus.threads:
        key = f"P{t.proc_id}"
        if key in raw:
            instrs, offsets = _extract_one_function(raw[key], len(t.params))
            instrs = _pin_value_registers(instrs, t)
            result[key] = (instrs, offsets)
        else:
            result[key] = ([], [])
    return result


# ---------------------------------------------------------------------------
# Emit AArch64 litmus
# ---------------------------------------------------------------------------

def _format_instr(instr: AsmInstr) -> str:
    if instr.label:
        return f"{instr.label}:"
    mn = instr.mnemonic.upper()
    ops = instr.operands
    if mn in ("DMB", "DSB"):
        return f"{mn} {ops.upper()}"
    ops = re.sub(r"\b([wxbhsdq]\d+|[wx]zr|sp)\b",
                 lambda m: m.group(0).upper(), ops, flags=re.IGNORECASE)
    ops = re.sub(r"\.LBB(\d+_\d+)", r"L\1", ops)
    return f"{mn} {ops}"


def _remap_labels(instrs: list, proc_id: int) -> list:
    label_map = {}
    counter = 0
    for ins in instrs:
        if ins.label and ins.label.startswith(".LBB"):
            label_map[ins.label] = f"L{proc_id}e{counter}"
            counter += 1
    result = []
    for ins in instrs:
        if ins.label:
            result.append(AsmInstr("", "", label=label_map.get(ins.label, ins.label)))
        else:
            ops = ins.operands
            for old, new in label_map.items():
                ops = ops.replace(old, new)
            result.append(AsmInstr(ins.mnemonic, ops))
    return result


def _reg_for_local(var_name: str, thread: Thread) -> str:
    try:
        idx = thread.local_vars.index(var_name)
    except ValueError:
        return None
    return f"X{19 + idx}" if idx < len(PIN_REGS) else None


def _translate_condition(litmus: CLitmus) -> str:
    cond = litmus.condition
    for t in litmus.threads:
        for rv in sorted(t.local_vars, key=len, reverse=True):
            reg = _reg_for_local(rv, t)
            if reg:
                cond = re.sub(rf"\b{t.proc_id}:{re.escape(rv)}\b",
                              f"{t.proc_id}:{reg}", cond)
    return cond


def emit_litmus(litmus: CLitmus, funcs: dict, variant: str) -> str:
    lines = [f"AArch64 {litmus.name}_{variant}", "{"]

    for t in litmus.threads:
        _, offsets = funcs.get(f"P{t.proc_id}", ([], []))
        if offsets:
            lines.append(f" int p{t.proc_id}s[{len(offsets)}];")

    for t in litmus.threads:
        for i, (_, pname) in enumerate(t.params):
            lines.append(f" {t.proc_id}:X{i}={pname};")
        _, offsets = funcs.get(f"P{t.proc_id}", ([], []))
        if offsets:
            lines.append(f" {t.proc_id}:X{len(t.params)}=p{t.proc_id}s;")
    lines.append("}")

    columns = []
    for t in litmus.threads:
        thread_instrs, _ = funcs.get(f"P{t.proc_id}", ([], []))
        thread_instrs = _remap_labels(thread_instrs, t.proc_id)
        columns.append([_format_instr(ins) for ins in thread_instrs])

    max_rows = max((len(c) for c in columns), default=0)
    for c in columns:
        c.extend([""] * (max_rows - len(c)))
    widths = [max(max((len(s) for s in c), default=2), 4) for c in columns]

    hdr = " | ".join(f"P{i:<{widths[i]-1}}" for i in range(len(columns)))
    lines.append(f" {hdr} ;")
    for row in range(max_rows):
        cells = " | ".join(f"{c[row]:<{widths[i]}}" for i, c in enumerate(columns))
        lines.append(f" {cells} ;")

    lines.append("")
    lines.append(_translate_condition(litmus))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# herd7
# ---------------------------------------------------------------------------

def _run_herd7(litmus_path: str, herd_cmd: str, cat_model: str, libdir: str) -> dict:
    cmd = f"{herd_cmd} -I {libdir} -model {cat_model} {litmus_path}"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    output = r.stdout + r.stderr
    obs = next((l for l in output.splitlines() if l.startswith("Observation")), None)
    return {"output": output.strip(), "observation": obs}


def _obs_str(result: dict) -> str:
    if "error" in result:
        return "ERR"
    obs = (result.get("herd", {}).get("observation", "")
           or result.get("observation", ""))
    if not obs:
        return "?"
    for tag in ("Never", "Sometimes", "Always"):
        if tag in obs:
            return tag
    return "?"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _compile_variant(litmus, c_source, compiler, herd_cmd, cat, libdir, outdir):
    try:
        asm = compile_to_asm(c_source, compiler)
    except RuntimeError as e:
        return {"error": str(e)}

    funcs = extract_functions(asm, litmus)
    if not funcs:
        return {"error": "No functions found in assembly"}

    text = emit_litmus(litmus, funcs, compiler.name)
    path = os.path.abspath(os.path.join(outdir, f"{litmus.name}_{compiler.name}.litmus"))
    Path(path).write_text(text)

    herd = {"output": "(not run)"}
    if herd_cmd:
        try:
            herd = _run_herd7(path, herd_cmd, cat, libdir)
        except Exception as e:
            herd = {"error": str(e)}
    return {"litmus_path": path, "herd": herd}


def process_one(c_litmus_path, compilers, herd_cmd, cat, libdir, outdir,
                save_c=False):
    litmus = parse_c_litmus(c_litmus_path)
    c_source = wrap_as_c(litmus)

    if save_c:
        Path(os.path.join(outdir, f"{litmus.name}.c")).write_text(c_source)

    c_ref = {}
    if herd_cmd:
        try:
            c_ref = _run_herd7(c_litmus_path, herd_cmd, "rc11.cat", libdir)
        except Exception as e:
            c_ref = {"error": str(e)}

    results = {cc.name: _compile_variant(litmus, c_source, cc,
                                         herd_cmd, cat, libdir, outdir)
               for cc in compilers}
    return {"name": litmus.name, "results": results, "c_ref": c_ref}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _classify(all_results, compilers):
    names = [c.name for c in compilers]
    ok = mm = err = 0
    for r in all_results:
        if "error" in r:
            err += 1
            continue
        obs = [_obs_str(r["results"].get(n, {})) for n in names]
        valid = [o for o in obs if o not in ("ERR", "?")]
        if not valid:
            err += 1
        elif len(set(valid)) == 1:
            ok += 1
        else:
            mm += 1
    return ok, mm, err


def write_csv(all_results, compilers, outdir):
    names = [c.name for c in compilers]
    csv_path = os.path.join(outdir, "litmus-diff.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "rc11"] + names + ["status"])
        for r in all_results:
            if "error" in r:
                w.writerow([r["name"], f'ERR: {r["error"]}'] + [""] * len(names) + ["ERR"])
                continue
            ref = _obs_str(r.get("c_ref", {}))
            obs = [_obs_str(r["results"].get(n, {})) for n in names]
            valid = [o for o in obs if o not in ("ERR", "?")]
            status = "ERR" if not valid else "OK" if len(set(valid)) == 1 else "MISMATCH"
            w.writerow([r["name"], ref] + obs + [status])
    return csv_path


def print_summary(all_results, compilers, outdir):
    ok, mm, err = _classify(all_results, compilers)
    csv_path = write_csv(all_results, compilers, outdir)
    print(f"\nTotal: {len(all_results)} tests, {ok} OK, {mm} MISMATCH, {err} ERR")
    print(f"Results written to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="C litmus -> AArch64 litmus diff checker")
    ap.add_argument("input", nargs="?", help="C litmus file")
    ap.add_argument("--batch", help="Directory of .litmus files")
    ap.add_argument("--config", help="Path to config.json")
    ap.add_argument("--clang", default="clang", help="Clang binary (no --config)")
    ap.add_argument("--herd7", default="")
    ap.add_argument("--cat", default="")
    ap.add_argument("--libdir", default="")
    ap.add_argument("--outdir", default="generated",
                    help="Output directory (default: generated/)")
    ap.add_argument("--save-c", action="store_true",
                    help="Save compilable C source for each test")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    if args.config:
        cfg = load_config(args.config)
        compilers = cfg["compilers"]
        if not compilers:
            sys.exit("No compilers in config")
        herd_cmd = args.herd7 or cfg["herd7"]
        cat = args.cat or cfg["cat"]
        libdir = args.libdir or cfg["libdir"]
        litmus_dir = cfg.get("litmus_tests", "")
    else:
        compilers = [CompilerConfig("clang", args.clang, ["-O3"]),
                     CompilerConfig("clangir", args.clang, ["-O3", "-fclangir"]),
                     CompilerConfig("orb", args.clang, ["-O3", "-fclangir", "-Xclang", "-orb"])]
        herd_cmd, cat, libdir, litmus_dir = args.herd7, args.cat, args.libdir, ""

    # Duplicates in c11popl15 catalogue (same test, different name)
    SKIP = {"b", "a5", "strengthen2"}

    if args.batch:
        inputs = sorted(Path(args.batch).glob("*.litmus"))
    elif args.input:
        inputs = [Path(args.input)]
    elif litmus_dir:
        inputs = sorted(Path(litmus_dir).glob("*.litmus"))
    else:
        ap.print_help()
        sys.exit(1)
    inputs = [p for p in inputs if p.stem not in SKIP]

    results = []
    for inp in inputs:
        try:
            results.append(process_one(str(inp), compilers, herd_cmd,
                                       cat, libdir, outdir, save_c=args.save_c))
        except Exception as e:
            results.append({"name": inp.stem, "error": str(e)})

    print_summary(results, compilers, outdir)


if __name__ == "__main__":
    main()
