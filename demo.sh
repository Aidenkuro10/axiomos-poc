#!/usr/bin/env bash
set -euo pipefail

CU_IN=${1:-examples/mini_vector_add.cu}

# 1) CUDA -> AXIR
python3 axiomify.py "$CU_IN" --emit-axir -o demo.axir.json

# 2) AXIR -> HIP (glue, à titre de preuve du pivot)
python3 axiomify.py demo.axir.json --from-axir -o demo_glue.hip.cpp

# 3) CUDA -> HIP (fichier complet lisible)
python3 axiomify.py "$CU_IN" --hip-direct -o demo_full.hip.cpp

# 4) HIP -> CUDA (pour exécution sur GPU NVIDIA/Colab)
python3 hip2cuda.py demo_full.hip.cpp demo_from_hip.cu

# 5) Compile & run (multi-arch pour éviter l’erreur PTX)
nvcc demo_from_hip.cu -o demo_exec \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89

./demo_exec
