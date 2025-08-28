#!/usr/bin/env python3
"""AXIOMify v0 — PoC traducteur CUDA -> HIP (AMD ROCm)
Usage:
  python3 axiomify.py path/to/input.cu > output_hip.cpp
Limites: remplacements basiques de l’API runtime (cuda* -> hip*). Le kernel reste inchangé.
"""
import re, sys, pathlib

CUDA_TO_HIP = {
    r'#\s*include\s*<cuda_runtime.h>': '#include <hip/hip_runtime.h>',
    r'\bcudaMalloc\b': 'hipMalloc',
    r'\bcudaFree\b': 'hipFree',
    r'\bcudaMemcpy\b': 'hipMemcpy',
    r'\bcudaMemcpyHostToDevice\b': 'hipMemcpyHostToDevice',
    r'\bcudaMemcpyDeviceToHost\b': 'hipMemcpyDeviceToHost',
    r'\bcudaMemcpyDeviceToDevice\b': 'hipMemcpyDeviceToDevice',
    r'\bcudaDeviceSynchronize\b': 'hipDeviceSynchronize',
    r'\bcudaGetLastError\b': 'hipGetLastError',
    r'\bcudaError_t\b': 'hipError_t',
    r'\bcudaSuccess\b': 'hipSuccess',
}

def hipify(text: str) -> str:
    out = text
    for pat, repl in CUDA_TO_HIP.items():
        out = re.sub(pat, repl, out)
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 axiomify.py input.cu > output.cpp", file=sys.stderr)
        sys.exit(1)
    src = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')
    sys.stdout.write(hipify(src))

if __name__ == '__main__':
    main()
