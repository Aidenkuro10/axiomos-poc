#!/usr/bin/env bash
set -euo pipefail

CU_IN=${1:-examples/mini_vector_add.cu}

echo "===== [1] Code CUDA original ====="
head -n 40 "$CU_IN"
echo

# 1) CUDA -> AXIR
python3 axiomify.py "$CU_IN" --emit-axir -o demo.axir.json
echo "===== [2] AXIR (pivot universel) ====="
cat demo.axir.json
echo

# 2) AXIR -> HIP (glue)
python3 axiomify.py demo.axir.json --from-axir -o demo_glue.hip.cpp
echo "===== [3] HIP (glue simplifiée) ====="
head -n 40 demo_glue.hip.cpp
echo

# 3) CUDA -> HIP (fichier complet lisible)
python3 axiomify.py "$CU_IN" --hip-direct -o demo_full.hip.cpp
echo "===== [4] HIP complet généré ====="
head -n 60 demo_full.hip.cpp
echo

# 4) HIP -> CUDA (pour GPU NVIDIA)
python3 hip2cuda.py demo_full.hip.cpp demo_from_hip.cu
echo "===== [5] CUDA régénéré ====="
head -n 40 demo_from_hip.cu
echo

# 5) Compile & run
echo "===== [6] Compilation & exécution ====="
nvcc demo_from_hip.cu -o demo_exec \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89

./demo_exec

