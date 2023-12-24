#include "gemm.h"
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
