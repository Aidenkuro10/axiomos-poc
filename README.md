About this Repository

This repository contains the first Proof of Concept (v0) of AXIOM OS:

axiomify.py → a Python script that automatically converts CUDA runtime API calls into their HIP/ROCm equivalents.

examples/mini_vector_add.cu → a simple CUDA kernel (vector addition).

docs/ → roadmap PDF and visual timeline.

 Goal of this PoC: demonstrate that automatic translation from CUDA to HIP is possible on simple cases.

 Quick Usage
 1. Translate a CUDA file to HIP
python3 axiomify.py examples/mini_vector_add.cu > mini_vector_add_hip.cpp

2. Compile & Run

On NVIDIA (CUDA):

nvcc examples/mini_vector_add.cu -o mini_vector_add
./mini_vector_add


On AMD (ROCm/HIP):

hipcc mini_vector_add_hip.cpp -o mini_vector_add_hip
./mini_vector_add_hip


💡 Note: The kernel launch syntax <<<...>>> is generally accepted by HIP, making portability easier.

📂 Repository Structure
axiomos-poc/
├─ axiomify.py
├─ examples/
│  ├─ mini_vector_add.cu
│  └─ README.md
├─ docs/
│  ├─ axiom_os_roadmap_visual.pdf
│  └─ axiom_os_timeline.png
├─ LICENSE
└─ .gitignore

Roadmap (excerpt)

Phase 0: CUDA → HIP PoC (this repo) 

Phase 1: Broader API coverage (streams/events, cuBLAS → hipBLAS, cuRAND → hipRAND)

Phase 2: Structured front-end (AST parser, internal IR)

Phase 3: Intel backend (SYCL/oneAPI)

Phase 4: Framework bridges (PyTorch ops)

Phase 5: Performance & auto-tuning

Phase 6: Pro Edition (Analyzer, Portability Score)

 See docs/axiom_os_roadmap_visual.pdf
 for details.

  Known Limitations (v0)

Simple text-based replacements (no syntax analysis).

Covers only a subset of the CUDA runtime API.

Libraries (cuBLAS/cuDNN) not handled yet.

Performance is not the goal of this PoC.

 License

MIT — free to use, modify, and redistribute.

 With this PoC, AXIOM OS takes its first concrete step toward becoming the universal AI operating system — the Windows of AI middleware.
