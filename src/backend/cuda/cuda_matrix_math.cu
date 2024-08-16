
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <backend/cuda/cuda_matrix_math.cuh>
#include <cuda_utils.h>


__global__ void BLAS2AddKernel(const float* a, const float* b, float* c, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		c[col * rows + row] = a[col * rows + row] + b[row];
	}
}


__global__ void BLAS3AddKernel(const float *a, const float *b, float *c, size_t cols, size_t rows) {
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
	
	checkCublasError(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

	return C;
}


__global__ void normalizeKernel(float* data, float* weights, float* bias, float* allocatedMem, size_t cols, size_t rows) {

	float* mean = allocatedMem;
	float* stddev = allocatedMem + rows;

	int col = threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < rows) {
		float sum = 0.0f;
		for (int i = col; i < cols; i += blockDim.x) {
			sum += data[i * rows + row];
		}

		atomicAdd(&mean[row], sum);
	}
	__syncthreads();

	if (row < rows && col == 0) {
		mean[row] /= rows;
	}
	__syncthreads();

	if (row < rows) {
		float sumSqDiff = 0.0f;
		for (int i = col; i < cols; i += blockDim.x) {
			float diff = data[i * rows + row] - mean[row];
			sumSqDiff += diff * diff;
		}

		atomicAdd(&stddev[row], sumSqDiff);
	}
	__syncthreads();

	if (row < rows && col == 0) {
		stddev[row] = sqrtf(stddev[row] / rows);
	}
	__syncthreads();

	if (row < rows) {
		for (int i = col; i < cols; i += blockDim.x) {
			data[i * rows + row] = ((data[i * rows + row] - mean[row]) / stddev[row]) * weights[row] + bias[row];
		}
	}
}

void cudaNormalizeMatrix(tfm::Tensor& matrix) {
	checkCudaError(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = matrix.cols();
	size_t rows = matrix.rows();

	tfm::Device device(tfm::DeviceType::CUDA, 0);
	matrix.moveTo(device);
	
	// Launch a kernel on the GPU with one thread for each element.
	dim3 blockSize(16, 16);
	dim3 gridSize((rows + blockSize.y - 1) / blockSize.y);

	float* mem = nullptr;
	checkCudaError(cudaMalloc((void**)mem, 2 * rows * sizeof(float)), "cudaMalloc failed");
	normalizeKernel<<<gridSize, blockSize>>>(matrix.data(), matrix.weights(), matrix.bias(), mem, cols, rows);

	// Check for any errors launching the kernel
	checkCudaError(cudaGetLastError(), "normalizeKernel launch failed");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

	return;
}


__global__ void ReLUKernel(float* data, size_t cols, size_t rows)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows && data[col * rows + row] < 0) {
		data[col * rows + row] = 0;
	}
}

void cudaReLU(tfm::Tensor& matrix) {
	checkCudaError(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = matrix.cols();
	size_t rows = matrix.rows();

	tfm::Device device(tfm::DeviceType::CUDA, 0);
	matrix.moveTo(device);
	
	// Launch a kernel on the GPU with one thread for each element.
	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	ReLUKernel<<<gridSize, blockSize>>>(matrix.data(), cols, rows);

	// Check for any errors launching the kernel
	checkCudaError(cudaGetLastError(), "ReLUKernel launch failed");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

	return;
}


