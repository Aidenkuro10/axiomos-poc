# AXIOM OS — PoC v0.3: CUDA → HIP (AMD ROCm) Translator

**Vision**: Free AI from the CUDA lock-in. AXIOM OS is a middleware that enables the same AI code to run across different processors (NVIDIA, AMD, Intel, and more) through automatic translation.

 Official website: [axiomos.ai](https://axiomos.ai)

---

 About this Repository

This repository contains the first Proof of Concept (v0.3) of AXIOM OS:

- **axiomify.py** → a Python script that automatically converts CUDA runtime API calls into their HIP/ROCm equivalents.
- **examples/** → sample CUDA kernels (vector addition, async/streams).
- **docs/** → roadmap PDF and visual timeline.

 **Goal of this PoC**: demonstrate that automatic translation from CUDA to HIP is possible, and provide basic coverage & reporting.

---

## Quick Usage

### 1. Report only (no file written)
```bash
python3 axiomify.py examples/mini_vector_add.cu --report-only

2. Translate one file → new file
python3 axiomify.py examples/mini_vector_add.cu -o mini_vector_add_hip.cpp

3. Translate one file in place
python3 axiomify.py examples/mini_vector_add.cu --inplace
# generates examples/mini_vector_add.hip.cpp

4. Bulk translate a directory
python3 axiomify.py ./examples -o ./examples_hip -v

Compile & Run

On NVIDIA (CUDA):

nvcc examples/mini_vector_add.cu -o mini_vector_add
./mini_vector_add


On AMD (ROCm/HIP):

hipcc mini_vector_add_hip.cpp -o mini_vector_add_hip
./mini_vector_add_hip


💡 Note: The kernel launch syntax <<<...>>> is generally accepted by HIP, making portability easier.

Repository Structure
axiomos-poc/
 ├─ axiomify.py
 ├─ examples/
 │   ├─ mini_vector_add.cu
 │   ├─ mini_async_copy.cu
 │   └─ README.md
 ├─ docs/
 │   ├─ axiom_os_roadmap_visual.pdf
 │   └─ axiom_os_timeline.png
 ├─ LICENSE
 └─ .gitignore

Roadmap (excerpt)

Phase 0: CUDA → HIP PoC (this repo) ✅

Phase 1: Broader API coverage (streams/events, cuBLAS → hipBLAS, cuRAND → hipRAND)

Phase 2: Structured front-end (AST parser, internal IR)

Phase 3: Intel backend (SYCL/oneAPI)

Phase 4: Framework bridges (PyTorch ops)

Phase 5: Performance & auto-tuning

Phase 6: Pro Edition (Analyzer, Portability Score)

See docs/axiom_os_roadmap_visual.pdf for details.

Known Limitations (v0.3)

Simple text-based replacements (no full syntax analysis).

Covers only a subset of the CUDA runtime API.

Libraries (cuBLAS/cuDNN) not handled yet.

Performance optimization is not the goal of this PoC.

License

MIT — free to use, modify, and redistribute.

 With this PoC, AXIOM OS takes its first concrete step toward becoming the universal AI operating system — the Windows of AI middleware.

