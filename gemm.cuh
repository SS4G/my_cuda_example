#include "gemm.h"
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define BLOCK_SIZE 16
// CUDA and CUBLAS functions
//#include <helper_functions.h>
//#include <helper_cuda.h>

__device__ void SetMatValGPU(Matrix *m, int rowIdx, int colIdx, float val) {
    m->data[rowIdx * (m -> width) + colIdx] = val;
    //m->data[0] = val;
}

__device__ float GetMatValGPU(Matrix *m, int rowIdx, int colIdx) {
    return m->data[rowIdx * (m -> width) + colIdx];
}

__global__ void GemmGPUFunc(Matrix* A, Matrix* B, Matrix* C) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("blockIdx.x=%d blockIdx.y=%d threadIdx.x=%d threadIdx.y=%d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    //printf("matIdx=(%02d, %02d)\n", i, k);
    //C->data[i * (C -> width) + k] = 1.0;
    //setMatVal(C, i, k, 5.0);
    if (i < A->height && k < B->width) {
        float res = 0;
        for (int j = 0 ; j < (A->width); j++) {
            //printf("A[%d][%d]=%f B[%d][%d]=%f\n", i, j, getMatVal(A, i, j), j, k, getMatVal(B, j, k));
            res += GetMatValGPU(A, i, j) * GetMatValGPU(B, j, k);
        }
        SetMatValGPU(C, i, k, res);
    }
}

// 使用shared memory 优化后的计算 大概加速8~10倍
// 具体参考 https://chiemon.github.io/2020/02/06/CUDA-%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95-%E4%BC%98%E5%8C%96%E5%8F%8A%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90-%E4%B8%8A.html
__global__ void GemmGPUSharedFunc(Matrix* A, Matrix* B, Matrix* C)
{
    int m = A -> height; // A = m x k
    int n = B -> width;  // B = k x n
    int k = A -> width;  // C = m x n
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float fCVal = 0.0f;

    __shared__ float shTileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shTileB[BLOCK_SIZE][BLOCK_SIZE];

    int nIter = (k + BLOCK_SIZE - 1) / BLOCK_SIZE; // k 维度上的迭代步数
    for(int i = 0; i < nIter; i++)
    {
        // load data from global memory to shared memory
        shTileA[threadIdx.y][threadIdx.x] = A->data[nRow * k + i * BLOCK_SIZE + threadIdx.x];
        shTileB[threadIdx.y][threadIdx.x] = B->data[(i * BLOCK_SIZE + threadIdx.y) * n + nCol];

        // sync to wait for all threads in one block to finish loading datas
        __syncthreads();

        // sub-matrix multiply
        #pragma unroll
        for(int l = 0; l < BLOCK_SIZE; l++)
        {
            fCVal += shTileA[threadIdx.y][l] * shTileB[l][threadIdx.x];
        }

        // sync to wait for all threads in one block to finish compute
        __syncthreads();
    }

    // store results into global memory
    C->data[nRow * n + nCol] = fCVal;
}

/*
 cublas 目前看来是不可用的 需要环境中有对应的 so文件
*/
/*
void GemmGPUCublasFunc(Matrix* cpuA, Matrix* cpuB, Matrix* cpuC) {
    //cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //    cublasOperation_t transa, cublasOperation_t transb,
    //    int m, int n, int k,
    //    const float *alpha,
    //    const float *A, int lda,
    //    const float *B, int ldb,
    //    const float *beta,
    //    float *C, int ldc
    //);
    // A m * k
    // B k * n
    // C m * n

    // allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size_A = cpuA->width * cpuA->height;
    size_t size_B = cpuB->width * cpuB->height;
    size_t size_C = cpuC->width * cpuC->height;

    size_t memsize_A = size_A * sizeof(float);
    size_t memsize_B = size_B * sizeof(float);
    size_t memsize_C = size_C * sizeof(float);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cudaMalloc((void **) &d_A, memsize_A);
    cudaMalloc((void **) &d_B, memsize_B);
    cudaMalloc((void **) &d_C, memsize_C);

    cudaMemcpy(d_A, cpuA->data, memsize_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, cpuB->data, memsize_B, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(
        handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        cpuA->height, cpuB->width, cpuA->width, 
        &alpha, d_A, cpuA->height, d_B, cpuA->width, &beta, d_C, cpuA->height);

    cudaMemcpy(cpuC->data, d_C, memsize_C, cudaMemcpyDeviceToHost);
}
*/
// use cublas
