# Exemples

## `mini_vector_add.cu` (CUDA)
Addition de 2 petits vecteurs sur GPU NVIDIA (CUDA).

### Compiler & exécuter (machine NVIDIA + CUDA Toolkit)
```bash
nvcc mini_vector_add.cu -o mini_vector_add
./mini_vector_add
```

### Traduire vers HIP (AMD ROCm) avec AXIOMify
```bash
python3 ../axiomify.py mini_vector_add.cu > mini_vector_add_hip.cpp
```

### Compiler & exécuter (machine AMD ROCm)
```bash
hipcc mini_vector_add_hip.cpp -o mini_vector_add_hip
./mini_vector_add_hip
```
