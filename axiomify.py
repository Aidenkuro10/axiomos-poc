#!/usr/bin/env python3
"""
AXIOMify v0.2 — CUDA -> HIP translator with simple compatibility report.

Usage:
  # translate a single file
  python3 axiomify.py examples/mini_vector_add.cu -o mini_vector_add_hip.cpp

  # just see a compatibility report (no output file written)
  python3 axiomify.py examples/mini_vector_add.cu --report-only

  # verbose (shows what was matched/replaced)
  python3 axiomify.py examples/mini_vector_add.cu -o out.cpp -v
"""
import re, sys, argparse, pathlib
from collections import Counter

# --- mappings: extend runtime coverage a bit ---
CUDA_TO_HIP = {
    r'#\s*include\s*<cuda_runtime\.h>': '#include <hip/hip_runtime.h>',

    r'\bcudaMalloc\b'              : 'hipMalloc',
    r'\bcudaFree\b'                : 'hipFree',
    r'\bcudaMemcpy\b'              : 'hipMemcpy',
    r'\bcudaMemcpyAsync\b'         : 'hipMemcpyAsync',

    r'\bcudaMemcpyHostToDevice\b'  : 'hipMemcpyHostToDevice',
    r'\bcudaMemcpyDeviceToHost\b'  : 'hipMemcpyDeviceToHost',
    r'\bcudaMemcpyDeviceToDevice\b': 'hipMemcpyDeviceToDevice',

    r'\bcudaDeviceSynchronize\b'   : 'hipDeviceSynchronize',
    r'\bcudaGetLastError\b'        : 'hipGetLastError',
    r'\bcudaGetErrorString\b'      : 'hipGetErrorString',

    # Streams
    r'\bcudaStream_t\b'            : 'hipStream_t',
    r'\bcudaStreamCreate\b'        : 'hipStreamCreate',
    r'\bcudaStreamDestroy\b'       : 'hipStreamDestroy',
    r'\bcudaStreamSynchronize\b'   : 'hipStreamSynchronize',

    # Events
    r'\bcudaEvent_t\b'             : 'hipEvent_t',
    r'\bcudaEventCreate\b'         : 'hipEventCreate',
    r'\bcudaEventRecord\b'         : 'hipEventRecord',
    r'\bcudaEventSynchronize\b'    : 'hipEventSynchronize',
    r'\bcudaEventDestroy\b'        : 'hipEventDestroy',

    # Error types
    r'\bcudaError_t\b'             : 'hipError_t',
    r'\bcudaSuccess\b'             : 'hipSuccess',
}

CUDA_TOKEN_PATTERN = re.compile(r'\bcuda[A-Za-z_0-9]*\b')  # to detect any other cuda* calls


def translate(text: str, verbose: bool=False):
    out = text
    replacements = Counter()
    for pat, repl in CUDA_TO_HIP.items():
        before = out
        out = re.sub(pat, repl, out)
        if out != before:
            replacements[pat] += len(re.findall(repl, out))  # rough count
    return out, replacements


def compatibility_report(src_text: str, dst_text: str):
    """Return (covered, uncovered, unique_uncovered_set)"""
    all_cuda_calls = CUDA_TOKEN_PATTERN.findall(src_text)
    # Remove tokens that are part of includes or comments? keep simple for now
    covered = []
    for pat, repl in CUDA_TO_HIP.items():
        # create a rough list of covered tokens by what we map
        m = re.findall(r'cuda[A-Za-z_0-9]*', pat)
        if m:
            covered.append(m[0])

    # tokens we saw in source
    seen = Counter(all_cuda_calls)

    # consider covered those that appear in mapping
    covered_set = set(covered)
    uncovered = []
    for tok, cnt in seen.items():
        if tok not in covered_set:
            uncovered.append((tok, cnt))

    return seen, uncovered


def main():
    ap = argparse.ArgumentParser(description="AXIOMify — CUDA -> HIP translator with report")
    ap.add_argument("input", help="Path to .cu file")
    ap.add_argument("-o", "--output", help="Output translated file (e.g., out.cpp)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    ap.add_argument("--report-only", action="store_true", help="Only show compatibility report, no file written")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    src = in_path.read_text(encoding="utf-8")

    # Do translation
    dst, repl_counts = translate(src, verbose=args.verbose)

    # Report
    seen, uncovered = compatibility_report(src, dst)
    total_cuda_refs = sum(seen.values())
    total_uncovered = sum(cnt for _, cnt in uncovered)
    covered_refs = total_cuda_refs - total_uncovered
    coverage_pct = (covered_refs / total_cuda_refs * 100.0) if total_cuda_refs else 100.0

    print("=== AXIOM OS — Translation Report ===")
    print(f"Input file        : {in_path}")
    print(f"CUDA refs found   : {total_cuda_refs}")
    print(f"Covered refs      : {covered_refs}")
    print(f"Uncovered refs    : {total_uncovered}")
    print(f"Estimated coverage: {coverage_pct:.1f}%")
    if uncovered:
        top = ", ".join(f"{tok} x{cnt}" for tok, cnt in sorted(uncovered, key=lambda x:-x[1])[:10])
        print(f"Uncovered (top)   : {top}")
    else:
        print("Uncovered (top)   : —")

    if args.report_only:
        return

    if not args.output:
        print("\n[INFO] No -o provided; printing translated code to stdout\n")
        sys.stdout.write(dst)
        return

    pathlib.Path(args.output).write_text(dst, encoding="utf-8")
    print(f"\n[OK] Wrote translated file: {args.output}")


if __name__ == "__main__":
    main()

