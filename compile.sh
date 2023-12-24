# /environment/miniconda3/bin/nvcc
# gcc -o output_file source_file.cu -I /path/to/cuda/include -L /path/to/cuda/lib -lcudart -arch=sm_<version> -D CUDA_ENABLED
nvcc -std=c++11 -g -G gemm.cu -o gemm # -lcublas -arch=sm_60 -rdc=true -lcublas_device -lcudadevrt
if test $? -eq 0;
then
./gemm
else
    echo "compile fial"
fi

#nvcc example2.cu -o vec_add.run
#nv-nsight-cu-cli ./prof_test.run
#./vec_add.run
