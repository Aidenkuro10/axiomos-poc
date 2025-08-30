# AXIOM OS — PoC v0.4: CUDA → AXIR → HIP Translator (Universal Pivot)

**Vision**  
Free AI from the CUDA lock-in.  
AXIOM OS introduces **AXIR**, a universal intermediate representation that lets the same AI code run across NVIDIA, AMD, Intel, and more.  

CUDA is just the first step: Axiomos is building the **universal AI operating system**, breaking hardware silos and enabling true portability.

 Official website: [axiomos.ai](https://axiomos.ai)

---

## About this Repository

This repository contains the **Proof of Concept v0.4** of AXIOM OS.

- `axiomify.py` → Python toolchain with two stages:
  - **Front-end**: CUDA → AXIR (JSON intermediate representation)
  - **Back-end**: AXIR → HIP (AMD ROCm), or other targets in the future
  - Shortcut: `--hip-direct` for CUDA → HIP directly

- `hip2cuda.py` → Reverse translator (HIP → CUDA) to run HIP-generated code on NVIDIA GPUs (Colab, local).

- `demo.sh` → End-to-end demo (CUDA → AXIR → HIP → CUDA → Run).

- `examples/` → Sample CUDA kernels (vector addition, async/streams).

- `docs/` → Roadmap PDF and visual timeline.

---

##  Quick Usage

### 1. CUDA → AXIR
```bash
python3 axiomify.py examples/mini_vector_add.cu --emit-axir -o mini_vector_add.axir.json

2. AXIR → HIP
python3 axiomify.py mini_vector_add.axir.json --from-axir -o mini_vector_add_glue.hip.cpp

3. CUDA → HIP (direct shortcut)
python3 axiomify.py examples/mini_vector_add.cu --hip-direct -o mini_vector_add_hip.cpp

 Showtime Demo (end-to-end)

For the full pipeline (CUDA → AXIR → HIP → back to CUDA → run):

chmod +x demo.sh
./demo.sh


This will display:

The original CUDA code

The AXIR pivot JSON

The HIP (glue and full)

The regenerated CUDA

The final execution result:

Résultat : 11 22 33 44 55

 Repository Structure
axiomos-poc/
 ├─ axiomify.py          # CUDA <-> AXIR <-> HIP
 ├─ hip2cuda.py          # HIP -> CUDA reconversion
 ├─ demo.sh              # End-to-end demo
 ├─ examples/
 │   ├─ mini_vector_add.cu
 │   ├─ mini_async_copy.cu
 │   └─ README.md
 ├─ docs/
 │   ├─ axiom_os_roadmap_visual.pdf
 │   └─ axiom_os_timeline.png
 ├─ LICENSE
 └─ .gitignore

🗺 Roadmap (excerpt)

Phase 0: CUDA → HIP PoC (regex-based) ✅

Phase 1: Broader API coverage (streams/events) ✅

Phase 2 (v0.4): AXIR intermediate representation introduced ✅

Phase 3: Intel backend (SYCL/oneAPI)

Phase 4: Framework bridges (PyTorch ops)

Phase 5: Performance & auto-tuning

Phase 6: Pro Edition (Analyzer, Portability Score)

See docs/axiom_os_roadmap_visual.pdf for details.

⚠️ Known Limitations (v0.4)

AXIR is still minimal (covers malloc, memcpy, kernel launch, sync, free).

Only basic CUDA runtime APIs are supported.

Libraries (cuBLAS/cuDNN) not yet handled.

Performance optimization is not the goal of this PoC.

 License

MIT — free to use, modify, and redistribute.

With this PoC, AXIOM OS takes its first concrete step toward becoming the universal AI operating system — the Windows of AI middleware.
