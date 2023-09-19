#include <iostream>
__global__ void MatAdd(float* A, float* B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    C[i] = A[i] + 2 * B[i];
        // 获取全局索引
    //int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    //int stride = blockDim.x * gridDim.x;
    //for (int i = index; i < n; i += stride)
    //{
    //    C[i] = A[i] + B[i];
    //}
}

void vecAddTest() {
    const int N = 1 << 20;
    const int BLOCK_SIZE = 256;
    float *matA_Buf, *matB_Buf, *matC_Buf;
    cudaMallocManaged((void**)&matA_Buf, N * sizeof(float));
    cudaMallocManaged((void**)&matB_Buf, N * sizeof(float));
    cudaMallocManaged((void**)&matC_Buf, N * sizeof(float));

    // 初始化内存
    for (int i = 0; i < N; i++) {
        matA_Buf[i] = 10.0;
        matB_Buf[i] = 23.0;
    }

    for (int i = 0; i < 10; i++) {
        std::cout << matA_Buf[i] << ",";
    }
    std::cout << std::endl;

    for (int i = 0; i < 10; i++) {
        std::cout << matB_Buf[i] << ",";
    }
    std::cout << std::endl;

    dim3 blockSize(256);
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x);// kernel调用
    MatAdd<<<numBlocks, blockSize>>>(matA_Buf, matB_Buf, matC_Buf);

    //dim3 threadsPerBlock(BLOCK_SIZE); 
    //dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    //dim3 threadsPerBlock(BLOCK_SIZE);
    //dim3 threadsPerBlock(BLOCK_SIZE);
    //dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);// kernel调用

    // 定义kernel的执行配置
    //dim3 blockSize(256);
    //dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    //MatAdd << < gridSize, blockSize >> >(matA_Buf, matB_Buf, matC_Buf, N);

    
    cudaDeviceSynchronize();

    // 读取结果
    for (int i = 0; i < 10; i++) {
        std::cout << matC_Buf[i] << ",";
    }
    std::cout << std::endl;


    // 在原来的数据里面进行读取就可以了。
    cudaFree(matA_Buf); // 释放内存
    cudaFree(matB_Buf); // 释放内存
    cudaFree(matC_Buf); // 释放内存
}

int main() {
    vecAddTest();
    return 0;
}