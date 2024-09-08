#include <cuda_runtime.h>
#include <cublas_v2.h>


void check_cuda_error(cudaError_t result, const char* msg);
void check_cublas_error(cublasStatus_t result, const char* msg);
