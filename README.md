AXIOM OS — PoC v0: CUDA → HIP (AMD ROCm) Translator

Vision: Free AI from the CUDA lock-in. AXIOM OS is a middleware that enables the same AI code to run across different processors (NVIDIA, AMD, Intel, and more) through automatic translation.

This repository contains the first Proof of Concept (v0):

axiomify.py: a Python script that automatically converts CUDA runtime API calls into their HIP/ROCm equivalents.

examples/mini_vector_add.cu: a simple CUDA kernel (vector addition).

docs/: roadmap PDF and visual timeline.

🎯 Goal of this PoC: show that automatic translation from CUDA to HIP is possible on simple cases.

🔧 Quick Usage

Translate a CUDA file to HIP

python3 axiomify.py examples/mini_vector_add.cu > mini_vector_add_hip.cpp


Compile & Run

On NVIDIA (CUDA):

nvcc examples/mini_vector_add.cu -o mini_vector_add
./mini_vector_add


On AMD (ROCm/HIP):

hipcc mini_vector_add_hip.cpp -o mini_vector_add_hip
./mini_vector_add_hip


Note: The kernel launch syntax <<<...>>> is generally accepted by HIP, making portability easier.
