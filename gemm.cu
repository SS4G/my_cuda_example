#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>

double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}

#define EPS 1e-4
typedef struct matrix
{
    size_t height;
    size_t width;    /* data */
    float* data; 
} Matrix;

/*
 * CPU 矩阵相关函数
 */ 

void CHECK_STATUS(bool checkState, const char* info) {
    if (!checkState) {
        printf("ERROR:%s", info);
        exit(-1);
    }
}

void SetMatValCPU(Matrix *m, int rowIdx, int colIdx, float val) {
    m->data[rowIdx * (m -> width) + colIdx] = val;
}

float GetMatValCPU(Matrix *m, int rowIdx, int colIdx) {
    return m->data[rowIdx * (m -> width) + colIdx];
}

void GemmCPU(Matrix* m1, Matrix *m2, Matrix *out) {
    CHECK_STATUS(m1->width == m2->height, "gemm invlid size");
    double start = cpuSecond();
    for (size_t i = 0; i < m1 -> height; i++) {
        for (size_t k = 0; k < m2 -> width; k++) {
            float sumVal = 0.0;
            for (size_t j = 0; j < m2 -> width; j++) {
                sumVal += GetMatValCPU(m1, i, j) * GetMatValCPU(m2, j, k);
            }
            SetMatValCPU(out, i, k, sumVal);
        }
    }
    double end = cpuSecond();
    std::cout << "GemmCPU elapse:" << end - start << std::endl;
}

Matrix* NewMatrixCPU(size_t height, size_t width) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    CHECK_STATUS(mat != nullptr, "malloc ERR");
    mat -> height = height;
    mat -> width = width;
    mat -> data = (float*)malloc(sizeof(float) * width * height);
    CHECK_STATUS(mat -> data != nullptr, "malloc data ERR");
    return mat;
}

void FreeMatrixCPU(Matrix* m1) {
    free(m1 -> data);
    free(m1);
}


void RandomFillMatrixCPU(Matrix* m1, float B = 100.0, float A = 0.0) {
    srand (static_cast <unsigned> (time(0)));
    for (size_t i = 0; i < m1 -> height; i++) {
        for (size_t j = 0; j < m1 -> width; j++) {
            float randomVal = A + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(B-A)));
            SetMatValCPU(m1, i, j, randomVal);
        }
    }
}


void NormalFillMatrixCPU(Matrix* m1) {
    for (size_t i = 0; i < m1 -> height; i++) {
        for (size_t j = 0; j < m1 -> width; j++) {
            SetMatValCPU(m1, i, j, i * m1 -> width * m1 -> height + j);
        }
    }
}

void PrintMatrix(Matrix* m1, std::string info) {
    std::cout << "==========" << info << "==============\n";
    std::cout << "np.array([\n";
    for (size_t i = 0; i < m1 -> height; i++) {
        std::cout << "[";
        for (size_t j = 0; j < m1 -> width; j++) {
            std::cout << GetMatValCPU(m1, i, j) << ",";
        }
        std::cout << "],\n";
    }
    std::cout << "])\n";
    std::cout << std::endl;
}

bool CompareMatrixCPU(Matrix* m1, Matrix* m2) {
    if (m1 -> height != m2 -> height || m1 -> width != m2 -> width) {
        return false;
    }
    float diff = 0.0;
    for (size_t i = 0; i < m1 -> height; i++) {
        for (size_t j = 0; j < m1 -> width; j++) {
            //std::cout << GetMatValCPU(m1, i, j) << " || "<< GetMatValCPU(m2, i, j) << std::endl;
            diff += abs(GetMatValCPU(m1, i, j) - GetMatValCPU(m2, i, j));
        }
    }
    std::cout << "diff value:" << diff << std::endl;
    return diff < EPS;
}


/*
 * GPU 矩阵相关函数
 */

Matrix* NewMatrixGPUFromCPU(Matrix* cpuMat) {
    Matrix* mat;
    float* d_data;
    CHECK_STATUS(cudaSuccess == cudaMalloc((void**)&mat, sizeof(Matrix)), "cuda mat malloc failed");
    size_t data_size = sizeof(float) * cpuMat->height * cpuMat->width;
    //std::cout << "data_size:" << data_size << std::endl;

    CHECK_STATUS(cudaSuccess == cudaMalloc((void**)&d_data, data_size), "cuda data malloc failed");
    CHECK_STATUS(cudaSuccess == cudaMemcpy(d_data, cpuMat->data, data_size, cudaMemcpyHostToDevice), "cuda copy failed");

    float* tmp_h_data = cpuMat -> data;
    cpuMat -> data = d_data;
    CHECK_STATUS(cudaSuccess == cudaMemcpy(mat, cpuMat, sizeof(Matrix), cudaMemcpyHostToDevice), "cuda copy failed");
    cpuMat -> data = tmp_h_data;

    return mat;
}

Matrix* NewMatrixGPU(size_t height, size_t width) {
    Matrix tmpHostMat;
    Matrix* tmpHostMatPtr = &tmpHostMat;

    tmpHostMatPtr -> height = height;
    tmpHostMatPtr -> width = width;
    tmpHostMatPtr -> data = nullptr;
    Matrix* mat;
    CHECK_STATUS(cudaSuccess == cudaMalloc((Matrix**)&mat, sizeof(Matrix)), "gpu malloc failed");
    size_t data_size = sizeof(float) * width * height;
    CHECK_STATUS(cudaSuccess == cudaMalloc((float**)&(tmpHostMatPtr -> data), data_size),  "gpu malloc data failed");
    CHECK_STATUS(cudaSuccess == cudaMemcpy(mat, tmpHostMatPtr, sizeof(Matrix), cudaMemcpyHostToDevice), "cuda copy failed");
    return mat;
}

Matrix* NewMatrixCPUFromGPU(Matrix* mGpu) {
    Matrix* mCpu = (Matrix*)malloc(sizeof(Matrix));// ? 这里可以直接这么拷贝吗?
    CHECK_STATUS(cudaSuccess == cudaMemcpy(mCpu, mGpu, sizeof(Matrix), cudaMemcpyDeviceToHost), "cuda gpu to cpu failed");
    std::cout << "cpu copy done: width=" << mCpu -> width << " height="<< mCpu -> height << std::endl;
    size_t dataSize = sizeof(float) * mCpu -> width * mCpu -> height;
    float* cpuData = (float*)malloc(dataSize);
    std::cout << "cpuData done" << std::endl;
    CHECK_STATUS(cudaSuccess == cudaMemcpy(cpuData, mCpu -> data, dataSize, cudaMemcpyDeviceToHost), "cuda cpu to gpu failed");
    mCpu -> data = cpuData;
    std::cout << "cudaMemcpy done" << std::endl;
    return mCpu;
}

void FreeMatrixGPU(Matrix* m1) {
    Matrix tmpFree;
    CHECK_STATUS(cudaSuccess == cudaMemcpy(&tmpFree, m1, sizeof(Matrix), cudaMemcpyDeviceToHost), "cuda gpu to cpu failed");
    cudaFree(tmpFree.data);
    cudaFree(m1);
}

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

void GemmGPU(Matrix* m1, Matrix *m2, Matrix *out) {
    //CHECK_STATUS(m1->width == m2->height, "gemm invlid size");
    int BLOCK_X = 32; int BLOCK_Y = 32;
    //int GRID_X = (m2 -> width + BLOCK_X - 1) / BLOCK_X; int GRID_Y = (m1 -> height + BLOCK_Y - 1) / BLOCK_Y;
    //printf("GRID_X=%d, GRID_Y=%d", GRID_X, GRID_Y);

    dim3 blocks(BLOCK_X, BLOCK_Y);
    dim3 grids(5, 5);// kernel调用

    double start = cpuSecond();
    GemmGPUFunc<<<grids, blocks>>>(m1, m2, out);
    cudaDeviceSynchronize();
    double end = cpuSecond();
    std::cout << "GemmGPU elapse:" << end - start << std::endl;
}

void GPUTest() {
    bool printDebugFlag = true;
    // mat1 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    Matrix* m1CPU = NewMatrixCPU(3, 3);
    Matrix* m2CPU = NewMatrixCPU(3, 3);
    Matrix* mResCPU = NewMatrixCPU(3, 3); // 标准答案
    //Matrix* mResCPU = NewMatrixCPU(3, 3); // 计算答案

    RandomFillMatrixCPU(m1CPU);
    RandomFillMatrixCPU(m2CPU);
    //RandomFillMatrixCPU();
    GemmCPU(m1CPU, m2CPU, mResCPU);

    Matrix* m1GPU = NewMatrixGPUFromCPU(m1CPU);
    //std::cout << "malloc gpu1" << std::endl;

    Matrix* m2GPU = NewMatrixGPUFromCPU(m2CPU);
    //std::cout << "malloc gpu2" << std::endl;

    Matrix* mResGPU = NewMatrixGPU(m1CPU->height, m2CPU->width); // 计算答案
    //std::cout << "malloc gpu3" << std::endl;

    GemmGPU(m1GPU, m2GPU, mResGPU);
    //std::cout << "GemmGPU done" << std::endl;

    Matrix* mResCPULoad = NewMatrixCPUFromGPU(mResGPU);

    if (printDebugFlag) {
        PrintMatrix(m1CPU, "m1CPU");
        PrintMatrix(m2CPU, "m2CPU");
        PrintMatrix(mResCPU, "mResCPU");
        PrintMatrix(mResCPULoad, "mResCPULoad");
    }
    
    bool cmpRes = CompareMatrixCPU(mResCPU, mResCPULoad);
    FreeMatrixCPU(m1CPU);
    FreeMatrixCPU(m2CPU);
    //FreeMatrixCPU(stdmRes);
    FreeMatrixCPU(mResCPU);
    FreeMatrixCPU(mResCPULoad);
    
    //std::cout << "cpu freed" << std::endl;
    FreeMatrixGPU(m1GPU);
    FreeMatrixGPU(m2GPU);
    FreeMatrixGPU(mResGPU);
    //std::cout << "gpu freed" << std::endl;

    if (cmpRes) {
        std::cout << "INFO: gemm Success" <<  std::endl;
    } else {
        std::cout << "ERROR: gemm Failed" << std::endl;
    }
    
}

int main() {
    GPUTest();
}