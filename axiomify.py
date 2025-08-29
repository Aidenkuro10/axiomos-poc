#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AXIOMify v0.4.3 — CUDA <-> AXIR <-> HIP translator

Modes:
  # CUDA -> AXIR
  python3 axiomify.py examples/vector_add.cu --emit-axir -o vector_add.axir.json

  # AXIR -> HIP (glue simplifiée)
  python3 axiomify.py vector_add.axir.json --from-axir -o vector_add_glue.hip.cpp

  # CUDA -> HIP direct (fichier complet lisible)
  python3 axiomify.py examples/vector_add.cu --hip-direct -o vector_add_hip.cpp

  # Chaîne 2 étapes
  python3 axiomify.py examples/vector_add.cu --emit-axir -o tmp.axir.json
  python3 axiomify.py tmp.axir.json --from-axir -o vector_add_glue.hip.cpp
"""

import re, os, sys, argparse, pathlib, json
from collections import Counter

# =========================================================
# Utilitaires
# =========================================================

# Tokens CUDA pour mini-rapport (facultatif)
CUDA_TOKEN_PATTERN = re.compile(r'\bcuda[A-Za-z_0-9]*\b')

def coverage_report(src: str, mapping_patterns):
    seen = Counter(CUDA_TOKEN_PATTERN.findall(src))
    covered_tokens = set()
    for pat in mapping_patterns:
        for tok in re.findall(r'cuda[A-Za-z_0-9]*', pat):
            covered_tokens.add(tok)
    uncovered = [(tok, cnt) for tok, cnt in seen.items() if tok not in covered_tokens]
    total = sum(seen.values())
    miss  = sum(cnt for _, cnt in uncovered)
    pct = 100.0 if total == 0 else (100.0 * (total - miss) / total)
    return pct, uncovered

# =========================================================
# FRONT-END: CUDA -> AXIR
# =========================================================

# Kernel launch avec args: foo<<<grid, block[, sharedMem][, stream]>>>(arg1, arg2, ...)
LAUNCH_RE = re.compile(
    r'([A-Za-z_]\w*)\s*<<<\s*'          # kernel name
    r'([^,>]+)\s*,\s*'                  # grid (1 composante ou expr)
    r'([^,>]+)'                         # block (1 composante ou expr)
    r'(?:\s*,\s*[^,>]+)?'               # optional sharedMem (ignoré)
    r'(?:\s*,\s*[^>]+)?'                # optional stream (ignoré)
    r'\s*>>>\s*\(\s*'                   # close <<< >>>
    r'([^)]*)\)'                        # args (peuvent être vides)
)

def parse_kernel_launches(src: str):
    launches = []
    for m in LAUNCH_RE.finditer(src):
        kernel = m.group(1).strip()
        grid_1 = m.group(2).strip()
        block_1 = m.group(3).strip()
        args_s = m.group(4).strip()
        args = [a.strip() for a in args_s.split(",")] if args_s else []
        launches.append({
            "op": "KernelLaunch",
            "kernel": kernel,
            "grid": [grid_1, "1", "1"],
            "block": [block_1, "1", "1"],
            "args": args
        })
    return launches

def cuda_to_axir(src: str):
    ops = []

    # cudaMalloc(&ptr, size)
    for m in re.finditer(r'cudaMalloc\(([^,]+),\s*([^)]+)\)', src):
        ops.append({"op": "DeviceMalloc", "dst": m.group(1).strip(), "bytes": m.group(2).strip()})

    # cudaMemcpy(dst, src, size, kind=H2D/D2H)
    for m in re.finditer(r'cudaMemcpy\(([^,]+),\s*([^,]+),\s*([^,]+),\s*(cudaMemcpyHostToDevice|cudaMemcpyDeviceToHost)\)', src):
        kind = "H2D" if "HostToDevice" in m.group(4) else "D2H"
        ops.append({"op": "Memcpy",
                    "dst": m.group(1).strip(),
                    "src": m.group(2).strip(),
                    "bytes": m.group(3).strip(),
                    "kind": kind})

    # Synchronisation
    if re.search(r'\bcudaDeviceSynchronize\b', src):
        ops.append({"op": "DeviceSynchronize"})

    # cudaFree(ptr)
    for m in re.finditer(r'cudaFree\(([^)]+)\)', src):
        ops.append({"op": "DeviceFree", "ptr": m.group(1).strip()})

    # Kernel launches
    ops.extend(parse_kernel_launches(src))

    return {"version": "0.1", "ops": ops}

# =========================================================
# BACK-END: AXIR -> HIP (glue)
# =========================================================

def axir_to_hip(axir):
    """
    Génère un 'glue' HIP minimal à partir des ops AXIR.
    Le but est de prouver la chaîne AXIR -> HIP (pas de reconstitution
    d'un programme complet ici).
    """
    lines = ["#include <hip/hip_runtime.h>", "// AXIR -> HIP (glue simplifiée)", ""]
    for op in axir.get("ops", []):
        t = op.get("op")
        if t == "DeviceMalloc":
            lines.append(f"hipMalloc({op['dst']}, {op['bytes']});")
        elif t == "Memcpy":
            kind = "hipMemcpyHostToDevice" if op["kind"] == "H2D" else "hipMemcpyDeviceToHost"
            lines.append(f"hipMemcpy({op['dst']}, {op['src']}, {op['bytes']}, {kind});")
        elif t == "DeviceSynchronize":
            lines.append("hipDeviceSynchronize();")
        elif t == "DeviceFree":
            lines.append(f"hipFree({op['ptr']});")
        elif t == "KernelLaunch":
            g = ",".join(op["grid"])
            b = ",".join(op["block"])
            args = ", ".join(op.get("args", []))
            lines.append(f"hipLaunchKernelGGL({op['kernel']}, dim3({g}), dim3({b}), 0, 0, {args});")
    lines.append("")
    return "\n".join(lines)

# =========================================================
# CUDA -> HIP direct (fichier complet lisible)
# =========================================================

CUDA_TO_HIP_PATTERNS = {
    r'#\s*include\s*<cuda_runtime\.h>' : '#include <hip/hip_runtime.h>',
    r'\bcudaMalloc\b'                  : 'hipMalloc',
    r'\bcudaFree\b'                    : 'hipFree',
    r'\bcudaMemset\b'                  : 'hipMemset',
    r'\bcudaMemcpyAsync\b'             : 'hipMemcpyAsync',
    r'\bcudaMemcpy\b'                  : 'hipMemcpy',
    r'\bcudaMemcpyHostToDevice\b'      : 'hipMemcpyHostToDevice',
    r'\bcudaMemcpyDeviceToHost\b'      : 'hipMemcpyDeviceToHost',
    r'\bcudaMemcpyDeviceToDevice\b'    : 'hipMemcpyDeviceToDevice',
    r'\bcudaDeviceSynchronize\b'       : 'hipDeviceSynchronize',
    r'\bcudaGetLastError\b'            : 'hipGetLastError',
    r'\bcudaPeekAtLastError\b'         : 'hipPeekAtLastError',
    r'\bcudaGetErrorString\b'          : 'hipGetErrorString',
    r'\bcudaError_t\b'                 : 'hipError_t',
    r'\bcudaSuccess\b'                 : 'hipSuccess',
    r'\bcudaStream_t\b'                : 'hipStream_t',
    r'\bcudaStreamCreateWithFlags\b'   : 'hipStreamCreateWithFlags',
    r'\bcudaStreamCreate\b'            : 'hipStreamCreate',
    r'\bcudaStreamSynchronize\b'       : 'hipStreamSynchronize',
    r'\bcudaStreamDestroy\b'           : 'hipStreamDestroy',
    r'\bcudaEvent_t\b'                 : 'hipEvent_t',
    r'\bcudaEventCreate\b'             : 'hipEventCreate',
    r'\bcudaEventRecord\b'             : 'hipEventRecord',
    r'\bcudaEventSynchronize\b'        : 'hipEventSynchronize',
    r'\bcudaEventElapsedTime\b'        : 'hipEventElapsedTime',
    r'\bcudaEventDestroy\b'            : 'hipEventDestroy',
}

LAUNCH_RE_DIRECT = re.compile(
    r'(\w+)\s*<<<\s*'          # kernel
    r'([^,>]+)\s*,\s*'         # grid
    r'([^,>]+)'                # block
    r'(?:\s*,\s*[^,>]+)?'      # sharedMem (opt)
    r'(?:\s*,\s*[^>]+)?'       # stream (opt)
    r'\s*>>>\s*\(\s*'          # >>>
    r'([^)]*)\)'               # args
)

def hip_direct_from_cuda(src: str):
    out = src
    for pat, repl in CUDA_TO_HIP_PATTERNS.items():
        out = re.sub(pat, repl, out)
    # Rewriter kernel launch -> hipLaunchKernelGGL
    def repl_launch(m):
        k = m.group(1)
        g = m.group(2).strip()
        b = m.group(3).strip()
        args = m.group(4).strip()
        # dim3(grid,1,1) si l'utilisateur n'a donné qu'une composante
        return f"hipLaunchKernelGGL({k}, dim3({g},1,1), dim3({b},1,1), 0, 0, {args})"
    out = LAUNCH_RE_DIRECT.sub(repl_launch, out)
    return out

# =========================================================
# CLI
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="AXIOMify v0.4.3 — CUDA <-> AXIR <-> HIP")
    ap.add_argument("input", help="Input file (.cu or .axir.json)")
    ap.add_argument("-o","--output", help="Output file")
    ap.add_argument("--emit-axir",  action="store_true", help="Translate CUDA -> AXIR (JSON)")
    ap.add_argument("--from-axir",  action="store_true", help="Translate AXIR -> HIP (glue)")
    ap.add_argument("--hip-direct", action="store_true", help="Translate CUDA -> HIP (full, readable)")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # CUDA -> AXIR
    if args.emit_axir:
        if in_path.suffix != ".cu":
            print(f"[ERROR] --emit-axir attend un .cu en entrée", file=sys.stderr)
            sys.exit(1)
        src = in_path.read_text(encoding="utf-8")
        axir = cuda_to_axir(src)
        out_path = pathlib.Path(args.output) if args.output else in_path.with_suffix(".axir.json")
        out_path.write_text(json.dumps(axir, indent=2))
        pct, uncovered = coverage_report(src, CUDA_TO_HIP_PATTERNS.keys())
        print(f"[OK] AXIR JSON écrit: {out_path} | Couverture approx CUDA*: {pct:.1f}%")
        if uncovered:
            top = ", ".join(f"{t} x{c}" for t,c in sorted(uncovered, key=lambda x:-x[1])[:8])
            print(f"[info] Tokens CUDA non mappés (top): {top}")
        return

    # AXIR -> HIP (glue)
    if args.from_axir:
        if not in_path.suffix.endswith(".json"):
            print(f"[ERROR] --from-axir attend un .json (AXIR) en entrée", file=sys.stderr)
            sys.exit(1)
        axir = json.loads(in_path.read_text(encoding="utf-8"))
        hip_code = axir_to_hip(axir)
        out_path = pathlib.Path(args.output) if args.output else in_path.with_suffix(".hip.cpp")
        out_path.write_text(hip_code)
        print(f"[OK] HIP (glue) écrit: {out_path}")
        return

    # CUDA -> HIP direct (lisible)
    if args.hip_direct:
        if in_path.suffix != ".cu":
            print(f"[ERROR] --hip-direct attend un .cu en entrée", file=sys.stderr)
            sys.exit(1)
        src = in_path.read_text(encoding="utf-8")
        hip_full = hip_direct_from_cuda(src)
        out_path = pathlib.Path(args.output) if args.output else in_path.with_suffix(".hip.cpp")
        out_path.write_text(hip_full)
        print(f"[OK] HIP (direct) écrit: {out_path}")
        return

    print("[ERROR] Choisis un mode: --emit-axir | --from-axir | --hip-direct", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()

