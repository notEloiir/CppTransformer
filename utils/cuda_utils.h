#include <cuda_runtime.h>
#include <cublas_v2.h>


void checkCudaError(cudaError_t result, const char* msg);
void checkCublasError(cublasStatus_t result, const char* msg);
