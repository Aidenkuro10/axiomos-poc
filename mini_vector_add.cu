// examples/mini_vector_add.cu — CUDA (NVIDIA)
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vectorAdd(const int* A, const int* B, int* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 5;
    int h_A[N] = {1, 2, 3, 4, 5};
    int h_B[N] = {10, 20, 30, 40, 50};
    int h_C[N] = {0, 0, 0, 0, 0};

    int *d_A=nullptr, *d_B=nullptr, *d_C=nullptr;
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));
    cudaMalloc(&d_C, N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    // 1 bloc, N threads
    vectorAdd<<<1, N>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Résultat : ");
    for (int i = 0; i < N; i++) printf("%d ", h_C[i]);
    printf("\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
