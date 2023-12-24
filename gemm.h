#ifndef __GEMM__
#define __GEMM__
typedef struct matrix
{
    size_t height;
    size_t width;    /* data */
    float* data; 
} Matrix;
#endif