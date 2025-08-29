#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AXIOMify v0.4 — CUDA <-> AXIR <-> HIP translator

Modes:
  # CUDA -> AXIR
  python3 axiomify.py examples/vector_add.cu --emit-axir -o vector_add.axir.json

  # AXIR -> HIP
  python3 axiomify.py vector_add.axir.json --from-axir -o vector_add_hip.cpp

  # CUDA -> AXIR -> HIP (2 étapes)
  python3 axiomify.py examples/vector_add.cu --emit-axir -o tmp.axir.json
  python3 axiomify.py tmp.axir.json --from-axir -o vector_add_hip.cpp
"""

import re, os, sys, argparse, pathlib, json
from collections import Counter

# =========================================================
# FRONT-END: CUDA -> AXIR
# =========================================================
def cuda_to_axir(src: str):
    ops = []

    # cudaMalloc(&ptr, size)
    for m in re.finditer(r'cudaMalloc\(([^,]+),\s*([^)]+)\)', src):
        ops.append({"op":"DeviceMalloc","dst":m.group(1).strip(),"bytes":m.group(2).strip()})

    # cudaMemcpy(dst, src, size, kind)
    for m in re.finditer(r'cudaMemcpy\(([^,]+),\s*([^,]+),\s*([^,]+),\s*(cudaMemcpyHostToDevice|cudaMemcpyDeviceToHost)\)', src):
        kind = "H2D" if "HostToDevice" in m.group(4) else "D2H"
        ops.append({"op":"Memcpy","dst":m.group(1).strip(),"src":m.group(2).strip(),"bytes":m.group(3).strip(),"kind":kind})

    # Kernel launch <<<grid,block>>>
    for m in re.finditer(r'(\w+)\s*<<<([^,]+),\s*([^>]+)>>>', src):
        ops.append({"op":"KernelLaunch","kernel":m.group(1),
                    "grid":[m.group(2).strip(),"1","1"],
                    "block":[m.group(3).strip(),"1","1"],
                    "args":[]})  # TODO: args parsing simplifié

    # cudaDeviceSynchronize
    if "cudaDeviceSynchronize" in src:
        ops.append({"op":"DeviceSynchronize"})

    # cudaFree(ptr)
    for m in re.finditer(r'cudaFree\(([^)]+)\)', src):
        ops.append({"op":"DeviceFree","ptr":m.group(1).strip()})

    return {"version":"0.1","ops":ops}

# =========================================================
# BACK-END: AXIR -> HIP
# =========================================================
def axir_to_hip(axir):
    lines = ["#include <hip/hip_runtime.h>", ""]
    for op in axir["ops"]:
        if op["op"] == "DeviceMalloc":
            lines.append(f"hipMalloc({op['dst']}, {op['bytes']});")
        elif op["op"] == "Memcpy":
            kind = "hipMemcpyHostToDevice" if op["kind"]=="H2D" else "hipMemcpyDeviceToHost"
            lines.append(f"hipMemcpy({op['dst']}, {op['src']}, {op['bytes']}, {kind});")
        elif op["op"] == "DeviceSynchronize":
            lines.append("hipDeviceSynchronize();")
        elif op["op"] == "DeviceFree":
            lines.append(f"hipFree({op['ptr']});")
        elif op["op"] == "KernelLaunch":
            lines.append(f"hipLaunchKernelGGL({op['kernel']}, "
                         f"dim3({op['grid'][0]},{op['grid'][1]},{op['grid'][2]}), "
                         f"dim3({op['block'][0]},{op['block'][1]},{op['block'][2]}), "
                         f"0,0);")
    return "\n".join(lines)

# =========================================================
# CLI
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="AXIOMify v0.4 — CUDA <-> AXIR <-> HIP")
    ap.add_argument("input", help="Input file (.cu or .axir.json)")
    ap.add_argument("-o","--output", help="Output file")
    ap.add_argument("--emit-axir", action="store_true", help="Translate CUDA -> AXIR (JSON)")
    ap.add_argument("--from-axir", action="store_true", help="Translate AXIR -> HIP")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # --- CUDA -> AXIR ---
    if args.emit_axir:
        src = in_path.read_text(encoding="utf-8")
        axir = cuda_to_axir(src)
        out_path = pathlib.Path(args.output) if args.output else in_path.with_suffix(".axir.json")
        out_path.write_text(json.dumps(axir, indent=2))
        print(f"[OK] AXIR JSON écrit: {out_path}")
        return

    # --- AXIR -> HIP ---
    if args.from_axir:
        axir = json.loads(in_path.read_text(encoding="utf-8"))
        hip_code = axir_to_hip(axir)
        out_path = pathlib.Path(args.output) if args.output else in_path.with_suffix(".hip.cpp")
        out_path.write_text(hip_code)
        print(f"[OK] HIP code écrit: {out_path}")
        return

    print("[ERROR] Choisissez --emit-axir ou --from-axir", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
