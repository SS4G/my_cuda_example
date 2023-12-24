# /environment/miniconda3/bin/nvcc
# gcc -o output_file source_file.cu -I /path/to/cuda/include -L /path/to/cuda/lib -lcudart -arch=sm_<version> -D CUDA_ENABLED
nvcc -g -G gemm.cu -o gemm
if test $? -eq 0;
then
./gemm
else
    echo "compile fial"
fi

#nvcc example2.cu -o vec_add.run
#nv-nsight-cu-cli ./prof_test.run
#./vec_add.run
