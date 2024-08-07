
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include "cuda_utils.h"


void checkCudaError(cudaError_t result, const char* msg) {
	if (result != cudaSuccess) {
		fprintf(stderr, "%s: %s", msg, cudaGetErrorString(result));
		exit(1);
	}
}


void checkCublasError(cublasStatus_t result, const char* msg) {
	if (result != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "%s: %s", msg, result);
		exit(1);
	}
}
