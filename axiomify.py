#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AXIOMify v0.3 — CUDA -> HIP translator with compatibility report
- Single file or whole directory
- --inplace support
- Simple coverage report

Usage examples:
  # Report only (no file written)
  python3 axiomify.py examples/mini_vector_add.cu --report-only

  # Translate one file -> out.cpp
  python3 axiomify.py examples/mini_vector_add.cu -o mini_vector_add_hip.cpp

  # Translate one file in place (creates file.hip.cpp next to it)
  python3 axiomify.py examples/mini_vector_add.cu --inplace

  # Bulk translate an entire directory -> output dir
  python3 axiomify.py ./examples -o ./examples_hip -v
"""
import re
import os
import sys
import argparse
import pathlib
from collections import Counter

# --- CUDA -> HIP mappings (extendable) ---
CUDA_TO_HIP = {
    # Includes
    r'#\s*include\s*<cuda_runtime\.h>': '#include <hip/hip_runtime.h>',

    # Memory & device management
    r'\bcudaMalloc\b'               : 'hipMalloc',
    r'\bcudaFree\b'                 : 'hipFree',
    r'\bcudaMemset\b'               : 'hipMemset',
    r'\bcudaGetDeviceCount\b'       : 'hipGetDeviceCount',
    r'\bcudaSetDevice\b'            : 'hipSetDevice',

    # Memcpy variants
    r'\bcudaMemcpyAsync\b'          : 'hipMemcpyAsync',
    r'\bcudaMemcpy\b'               : 'hipMemcpy',
    r'\bcudaMemcpyHostToDevice\b'   : 'hipMemcpyHostToDevice',
    r'\bcudaMemcpyDeviceToHost\b'   : 'hipMemcpyDeviceToHost',
    r'\bcudaMemcpyDeviceToDevice\b' : 'hipMemcpyDeviceToDevice',

    # Sync & errors
    r'\bcudaDeviceSynchronize\b'    : 'hipDeviceSynchronize',
    r'\bcudaGetLastError\b'         : 'hipGetLastError',
    r'\bcudaPeekAtLastError\b'      : 'hipPeekAtLastError',
    r'\bcudaGetErrorString\b'       : 'hipGetErrorString',
    r'\bcudaError_t\b'              : 'hipError_t',
    r'\bcudaSuccess\b'              : 'hipSuccess',

    # Streams
    r'\bcudaStream_t\b'             : 'hipStream_t',
    r'\bcudaStreamCreateWithFlags\b': 'hipStreamCreateWithFlags',
    r'\bcudaStreamCreate\b'         : 'hipStreamCreate',
    r'\bcudaStreamSynchronize\b'    : 'hipStreamSynchronize',
    r'\bcudaStreamDestroy\b'        : 'hipStreamDestroy',

    # Events
    r'\bcudaEvent_t\b'              : 'hipEvent_t',
    r'\bcudaEventCreate\b'          : 'hipEventCreate',
    r'\bcudaEventRecord\b'          : 'hipEventRecord',
    r'\bcudaEventSynchronize\b'     : 'hipEventSynchronize',
    r'\bcudaEventElapsedTime\b'     : 'hipEventElapsedTime',
    r'\bcudaEventDestroy\b'         : 'hipEventDestroy',
}

# Detect any cuda* tokens (rough) for coverage reporting
CUDA_TOKEN_PATTERN = re.compile(r'\bcuda[A-Za-z_0-9]*\b')

def translate(text: str, verbose: bool = False):
    """Apply regex mappings and return (translated_text, replacement_counts)."""
    out = text
    replacements = Counter()
    for pat, repl in CUDA_TO_HIP.items():
        out, n = re.subn(pat, repl, out)
        if n and verbose:
            print(f"[map] {pat} -> {repl}  (x{n})")
        if n:
            replacements[pat] += n
    return out, replacements

def compatibility_report(src_text: str, dst_text: str):
    """
    Compute a simple coverage report:
      - seen: Counter of all cuda* tokens in source
      - uncovered: list[(token, count)] of those not covered by our mapping
    """
    seen = Counter(CUDA_TOKEN_PATTERN.findall(src_text))

    # tokens considered "covered" (derived from our mapping keys)
    covered_tokens = set()
    for pat in CUDA_TO_HIP.keys():
        m = re.findall(r'\b(cuda[A-Za-z_0-9]*)\b', pat)
        # collect all cuda* tokens present in mapping patterns
        for tok in m:
            covered_tokens.add(tok)

    uncovered = [(tok, cnt) for tok, cnt in seen.items() if tok not in covered_tokens]
    return seen, uncovered

def translate_file(in_path: pathlib.Path, out_path: pathlib.Path, verbose: bool = False):
    src = in_path.read_text(encoding="utf-8")
    dst, _ = translate(src, verbose=verbose)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dst, encoding="utf-8")

def iter_cu_files(root: pathlib.Path):
    """Yield all .cu files under root (or the file itself if it's a .cu)."""
    if root.is_file():
        if root.suffix == ".cu":
            yield root
        return
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".cu"):
                yield pathlib.Path(dirpath) / f

def print_report(in_path: pathlib.Path, src: str, dst: str):
    seen, uncovered = compatibility_report(src, dst)
    total_cuda_refs = sum(seen.values())
    total_uncovered = sum(cnt for _, cnt in uncovered)
    covered_refs = total_cuda_refs - total_uncovered
    coverage_pct = (covered_refs / total_cuda_refs * 100.0) if total_cuda_refs else 100.0

    print("=== AXIOM OS — Translation Report ===")
    print(f"Input             : {in_path}")
    print(f"CUDA refs found   : {total_cuda_refs}")
    print(f"Covered refs      : {covered_refs}")
    print(f"Uncovered refs    : {total_uncovered}")
    print(f"Estimated coverage: {coverage_pct:.1f}%")
    if uncovered:
        top = ", ".join(f"{tok} x{cnt}" for tok, cnt in sorted(uncovered, key=lambda x: -x[1])[:10])
        print(f"Uncovered (top)   : {top}")
    else:
        print("Uncovered (top)   : —")

def main():
    ap = argparse.ArgumentParser(description="AXIOMify — CUDA -> HIP translator with report")
    ap.add_argument("input", help="Path to a .cu file OR a directory containing .cu files")
    ap.add_argument(
        "-o", "--output",
        help="Output file (for single input file) OR output directory (for directory input)"
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose mapping logs")
    ap.add_argument("--report-only", action="store_true", help="Show coverage report only (no file written)")
    ap.add_argument("--inplace", action="store_true", help="Translate in place (file.cu -> file.hip.cpp)")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # ----- Single file mode -----
    if in_path.is_file():
        if in_path.suffix != ".cu":
            print(f"[ERROR] Input file must end with .cu: {in_path}", file=sys.stderr)
            sys.exit(1)

        src = in_path.read_text(encoding="utf-8")
        dst, _ = translate(src, verbose=args.verbose)

        # Always show report
        print_report(in_path, src, dst)

        if args.report_only:
            return

        if args.inplace and not args.output:
            out_path = in_path.with_suffix(".hip.cpp")
            out_path.write_text(dst, encoding="utf-8")
            print(f"\n[OK] Wrote translated file: {out_path}")
            return

        if args.output:
            out_path = pathlib.Path(args.output)
            out_path.write_text(dst, encoding="utf-8")
            print(f"\n[OK] Wrote translated file: {out_path}")
            return

        # default: print to stdout
        print("\n[INFO] No -o provided; printing translated code to stdout\n")
        sys.stdout.write(dst)
        return

    # ----- Directory (bulk) mode -----
    out_dir = pathlib.Path(args.output) if args.output else (in_path / "_axiomified")
    translated = 0
    for cu_file in iter_cu_files(in_path):
        rel = cu_file.relative_to(in_path)
        out_path = out_dir / rel.with_suffix(".hip.cpp")
        translate_file(cu_file, out_path, verbose=args.verbose)
        translated += 1
        if args.verbose:
            print(f"[OK] {cu_file} -> {out_path}")

    print(f"\n[OK] Translated {translated} file(s) into: {out_dir}")

if __name__ == "__main__":
    main()
