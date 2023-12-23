#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>

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

void CHECK(bool checkState, const char* info) {
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
    CHECK(m1->width == m2->height, "gemm invlid size");
    for (size_t i = 0; i < m1 -> height; i++) {
        for (size_t k = 0; k < m2 -> width; k++) {
            float sumVal = 0.0;
            for (size_t j = 0; j < m2 -> width; j++) {
                sumVal += GetMatValCPU(m1, i, j) * GetMatValCPU(m2, j, k);
            }
            SetMatValCPU(out, i, k, sumVal);
        }
    }
}

Matrix* NewMatrixCPU(int width, int height) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    CHECK(mat != nullptr, "malloc ERR");
    mat -> height = height;
    mat -> width = width;
    mat -> data = (float*)malloc(sizeof(float) * width * height);
    CHECK(mat -> data != nullptr, "malloc data ERR");
    return mat;
}

void FreeMatrixCPU(Matrix* m1) {
    free(m1 -> data);
    free(m1);
}


void RandomFillMatrixCPU(Matrix* m1, float B = 100000.0, float A = -100000.0) {
    srand (static_cast <unsigned> (time(0)));
    for (size_t i = 0; i < m1 -> height; i++) {
        for (size_t j = 0; j < m1 -> width; j++) {
            float randomVal = A + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(B-A)));
            SetMatValCPU(m1, i, j, randomVal);
        }
    }
}

void VectorFillMatrixCPU(Matrix* m1, std::vector<std::vector<float>> initVec) {
    CHECK(initVec.size() == m1 -> height, "VectorFillMatrixCPU: height ERR");
    CHECK(initVec[0].size() == m1 -> width, "VectorFillMatrixCPU: width ERR");

    for (size_t i = 0; i < m1 -> height; i++) {
        for (size_t j = 0; j < m1 -> width; j++) {
            SetMatValCPU(m1, i, j, initVec[i][j]);
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
    std::cout << "==========" << info << "\n";
    for (size_t i = 0; i < m1 -> height; i++) {
        for (size_t j = 0; j < m1 -> width; j++) {
            std::cout << GetMatValCPU(m1, i, j) << ",";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

bool CompareMatrix(Matrix* m1, Matrix* m2) {
    if (m1 -> height != m2 -> height || m1 -> width != m2 -> width) {
        return false;
    }
    float diff = 0.0;
    for (size_t i = 0; i < m1 -> height; i++) {
        for (size_t j = 0; j < m1 -> width; j++) {
            diff += abs(GetMatValCPU(m1, i, j) - GetMatValCPU(m2, i, j));
        }
    }
    std::cout << "diff value:" << diff << std::endl;
    return diff < EPS;
}

// GPU 
/*
__device__ void setMatVal(Matrix *m, int rowIdx, int colIdx, float val) {
    m->data[rowIdx * (m -> width) + colIdx] = val;
    //m->data[0] = val;
}

__device__ float getMatVal(Matrix *m, int rowIdx, int colIdx) {
    return m->data[rowIdx * (m -> width) + colIdx];
}*/

int main() {
    bool printDebugFlag = false;
    // mat1 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    std::vector<std::vector<float>> m1Vals = {
        {1, 2, 3},
        {1, 2, 3},
        {1, 2, 3}
    };
    // mat2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    std::vector<std::vector<float>> m2Vals = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    std::vector<std::vector<float>> stdResVals = {
       {30, 36, 42},
       {30, 36, 42},
       {30, 36, 42}
    };
    Matrix* m1 = NewMatrixCPU(1000, 1000);
    Matrix* m2 = NewMatrixCPU(1000, 1000);
    Matrix* stdmRes = NewMatrixCPU(3, 3); // 标准答案
    Matrix* mRes = NewMatrixCPU(1000, 1000); // 计算答案

    //VectorFillMatrixCPU(m1, m1Vals);
    //VectorFillMatrixCPU(m2, m2Vals);
    //VectorFillMatrixCPU(stdmRes, stdResVals);

    double start = cpuSecond();
    GemmCPU(m1, m2, mRes);
    double end = cpuSecond();
    std::cout << "elapse:" << end - start << std::endl;

    if (printDebugFlag) {
        PrintMatrix(m1, "m1");
        PrintMatrix(m2, "m2");
        PrintMatrix(stdmRes, "stdmRes");
        PrintMatrix(mRes, "mRes");
    }

    FreeMatrixCPU(m1);
    FreeMatrixCPU(m2);
    FreeMatrixCPU(stdmRes);
    FreeMatrixCPU(mRes);
    bool cmpRes = CompareMatrix(stdmRes, mRes);
    if (cmpRes) {
        std::cout << "INFO: gemm Correct" <<  std::endl;
    } else {
        std::cout << "INFO: gemm Incorrect" << std::endl;
    }
}