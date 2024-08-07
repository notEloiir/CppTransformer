
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <backend/cuda/cuda_matrix_math.cuh>
#include <cuda_utils.h>


__global__ void BLAS2AddKernel(const float* a, const float* b, float* c, size_t cols, size_t rows)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		c[col * rows + row] = a[col * rows + row] + b[row];
	}
}


__global__ void BLAS3AddKernel(const float *a, const float *b, float *c, size_t cols, size_t rows)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		c[col * rows + row] = a[col * rows + row] + b[col * rows + row];
	}
}


// Helper function for using CUDA to add matrices in parallel.
tfm::Tensor cudaMatAdd(const tfm::Tensor& A, const tfm::Tensor& B) {
	checkCudaError(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = A.cols() < B.cols() ? A.cols() : B.cols();
	size_t rows = A.rows() < B.rows() ? A.rows() : B.rows();

	if (B.isVector()) {
		cols = A.cols();
	}

	tfm::Device device(tfm::DeviceType::CUDA, 0);
	const_cast<tfm::Tensor&>(A).moveTo(device);
	const_cast<tfm::Tensor&>(B).moveTo(device);
	tfm::Tensor C(cols, rows, device);

	// Launch a kernel on the GPU with one thread for each element.
	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	if (B.isVector()) {
		BLAS2AddKernel<<<gridSize, blockSize>>>(A.data(), B.data(), C.data(), cols, rows);
	}
	else {
		BLAS3AddKernel<<<gridSize, blockSize>>>(A.data(), B.data(), C.data(), cols, rows);
	}

	// Check for any errors launching the kernel
	checkCudaError(cudaGetLastError(), "BLAS3AddKernel launch failed");
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

	const_cast<tfm::Tensor&>(A).moveTo(tfm::Device(tfm::DeviceType::CPU));
	const_cast<tfm::Tensor&>(B).moveTo(tfm::Device(tfm::DeviceType::CPU));
	C.moveTo(tfm::Device(tfm::DeviceType::CPU));

	return C;
}


tfm::Tensor cudaMatMult(const tfm::Tensor& A, const tfm::Tensor& B, bool transposeA, bool transposeB) {
	cublasHandle_t handle;
	checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");
	
	cublasOperation_t transA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t transB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
	
	size_t m = !transposeA ? A.rows() : A.cols();
	size_t n = !transposeB ? B.cols() : B.rows();
	size_t k = !transposeA ? A.cols() : A.rows();

	const float alpha = 1.0f;
	const float beta = 0.0f;

	tfm::Device device(tfm::DeviceType::CUDA, 0);
	const_cast<tfm::Tensor&>(A).moveTo(device);
	const_cast<tfm::Tensor&>(B).moveTo(device);
	tfm::Tensor C(n, m, device);

	checkCublasError(
		cublasSgemm(handle,
			transA, transB,
			m, n, k,
			&alpha,
			A.data(), A.rows(),
			B.data(), B.rows(),
			&beta,
			C.data(), C.rows()),
		"Failed to perform matrix multiplication");

	const_cast<tfm::Tensor&>(A).moveTo(tfm::Device(tfm::DeviceType::CPU));
	const_cast<tfm::Tensor&>(B).moveTo(tfm::Device(tfm::DeviceType::CPU));
	C.moveTo(tfm::Device(tfm::DeviceType::CPU));
	
	checkCublasError(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

	return C;
}


void cudaNormalizeMatrix(tfm::Tensor& matrix) {
	// TODO: implement
	fprintf(stderr, "cudaNormalizeMatrix not implemented");
	exit(1);
}

void cudaReLU(tfm::Tensor& matrix) {
	// TODO: implement
	fprintf(stderr, "cudaReLU not implemented");
	exit(1);
}


