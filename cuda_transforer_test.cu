#include <iostream>

template <typename T>
typedef struct matrix
{
    int width;    /* data */
    int height;
    T* data; 
} Matrix;

/*
 *矩阵相关函数
 * 
 */
void InitMatrix(Matrix **matp, int height, int width) {
    cudaMallocManaged((void**)matp, sizeof(Matrix));
    Matrix *mat = *matp;
    mat -> width = width;  mat -> height = height;
    cudaMallocManaged((void**)&(mat->data), (mat->width) * (mat->height) * sizeof(float));
    return;
}

void DisplayMatrix(Matrix* mat) {
    std::cout << "=======" << std::endl;
    for (int i = 0; i < mat -> height; i++) {
        for (int j = 0; j < mat -> width; j++) {
            std::cout << mat -> data[i * mat -> width + j] << ",";
        }
        std::cout << std::endl;
    }
}

void FreeMatrix(Matrix* mat) {
    cudaFree(mat -> data); // 释放内存
    cudaFree(mat); // 释放内存
}

void GenerateRandomMatrix() {

}

void MatrixMultiplyTest() {
    const int I = 3;
    const int J = 4;
    const int K = 5;
    const int BLOCK_SIZE = 256;
    Matrix *matA, *matB, *matC;

    InitMatrix(&matA, I, J); FillMatrix(matA, 1.0);
    InitMatrix(&matB, J, K); FillMatrix(matB, 2.0);
    InitMatrix(&matC, I, K); FillMatrix(matC, 0.0);

    DisplayMatrix(matA);
    DisplayMatrix(matB);
    DisplayMatrix(matC);

    //std::cout << "init fin";
    int BLOCK_X = 32; int BLOCK_Y = 32;
    int GRID_X = (K + BLOCK_X - 1) / BLOCK_X; int GRID_Y = (I + BLOCK_Y - 1) / BLOCK_Y;
    printf("GRID_X=%d, GRID_Y=%d", GRID_X, GRID_Y);

    dim3 blocks(BLOCK_X, BLOCK_Y);
    dim3 grids(BATCH_SIZE, NUM_HEAD);// kernel调用
    MatMultiply<<<grids, blocks>>>(matA, matB, matC);
    
    // 必须有这句话
    cudaDeviceSynchronize();
    DisplayMatrix(matC);
    // 在原来的数据里面进行读取就可以了。
    FreeMatrix(matA); // 释放内存
    FreeMatrix(matB); // 释放内存
    FreeMatrix(matC); // 释放内存
}

int main() {
    MatrixMultiplyTest();
    return 0;
}