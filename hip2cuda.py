import re, sys, pathlib

def hip_to_cuda(text: str) -> str:
    # Remplacements fonctions + constantes
    text = (text
        .replace("#include <hip/hip_runtime.h>", "#include <cuda_runtime.h>")
        .replace("hipMalloc", "cudaMalloc")
        .replace("hipMemcpy", "cudaMemcpy")
        .replace("hipMemcpyHostToDevice", "cudaMemcpyHostToDevice")
        .replace("hipMemcpyDeviceToHost", "cudaMemcpyDeviceToHost")
        .replace("hipDeviceSynchronize", "cudaDeviceSynchronize")
        .replace("hipFree", "cudaFree")
        .replace("hipSuccess", "cudaSuccess")
        .replace("hipGetLastError", "cudaGetLastError")
        .replace("hipPeekAtLastError", "cudaPeekAtLastError")
        .replace("hipGetErrorString", "cudaGetErrorString")
    )

    # hipLaunchKernelGGL(...) -> kernel<<<dim3(G), dim3(B)>>>(args);
    pat = re.compile(
        r'hipLaunchKernelGGL\(\s*([A-Za-z_]\w*)\s*,\s*dim3\(([^)]*)\)\s*,\s*dim3\(([^)]*)\)\s*,\s*[^,]*\s*,\s*[^,]*\s*,\s*(.*?)\)\s*;',
        re.DOTALL
    )
    text = pat.sub(lambda m: f"{m.group(1)}<<<dim3({m.group(2)}), dim3({m.group(3)})>>>({m.group(4)});", text)
    return text

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 hip2cuda.py input.hip.cpp output.cu")
        sys.exit(1)
    inp, outp = map(pathlib.Path, sys.argv[1:3])
    src = inp.read_text()
    out = hip_to_cuda(src)
    outp.write_text(out)
    print(f"[OK] HIP -> CUDA : {inp} -> {outp}")

if __name__ == "__main__":
    main()
