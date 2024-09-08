
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <backend/cuda/cuda_matrix_math.cuh>
#include <cuda_utils.h>


__global__ void cuda_mat_add_BLAS2_kernel(const float* a, const float* b, float* c, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		c[col * rows + row] = a[col * rows + row] + b[row];
	}
}

__global__ void cuda_mat_add_BLAS3_kernel(const float *a, const float *b, float *c, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		c[col * rows + row] = a[col * rows + row] + b[col * rows + row];
	}
}

// Helper function for using CUDA to add matrices in parallel.
tfm::Tensor cuda_mat_add_BLAS3(const tfm::Tensor& A, const tfm::Tensor& B) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = A.cols() < B.cols() ? A.cols() : B.cols();
	size_t rows = A.rows() < B.rows() ? A.rows() : B.rows();

	if (B.is_vector()) {
		cols = A.cols();
	}

	tfm::Device device(tfm::DeviceType::CUDA, 0);
	const_cast<tfm::Tensor&>(A).move_to(device);
	const_cast<tfm::Tensor&>(B).move_to(device);
	tfm::Tensor C(cols, rows, device);

	// Launch a kernel on the GPU with one thread for each element.
	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	if (B.is_vector()) {
		cuda_mat_add_BLAS2_kernel<<<gridSize, blockSize>>>(A.data(), B.data(), C.data(), cols, rows);
	}
	else {
		cuda_mat_add_BLAS3_kernel<<<gridSize, blockSize>>>(A.data(), B.data(), C.data(), cols, rows);
	}

	// Check for any errors launching the kernel
	check_cuda_error(cudaGetLastError(), "cuda_mat_add_BLAS3_kernel launch failed");
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

	return C;
}


tfm::Tensor cuda_mat_add_along_axis(const tfm::Tensor& A, size_t axis) {
	// TODO: implement
	return tfm::Tensor();
}


tfm::Tensor cuda_mat_sub_BLAS3(const tfm::Tensor& A, const tfm::Tensor& B) {
	// TODO: implement
	return tfm::Tensor();
}


tfm::Tensor cuda_mat_mult_BLAS3(const tfm::Tensor& A, const tfm::Tensor& B, bool transpose_A, bool transpose_B) {
	cublasHandle_t handle;
	check_cublas_error(cublasCreate(&handle), "Failed to create cuBLAS handle");
	
	cublasOperation_t trans_A = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t trans_B = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;
	
	size_t m = !transpose_A ? A.rows() : A.cols();
	size_t n = !transpose_B ? B.cols() : B.rows();
	size_t k = !transpose_A ? A.cols() : A.rows();
	size_t k_check = !transpose_B ? B.rows() : B.cols();

	if (k != k_check) {
		char message[128];
		snprintf(message, 128, "Matrices have incompatible dimensions for multiplication: (%zu, %zu), (%zu, %zu)", k, m, n, k_check);
		throw std::runtime_error(message);
	}

	const float alpha = 1.0f;
	const float beta = 0.0f;

	tfm::Device device(tfm::DeviceType::CUDA, 0);
	const_cast<tfm::Tensor&>(A).move_to(device);
	const_cast<tfm::Tensor&>(B).move_to(device);
	tfm::Tensor C(n, m, device);

	check_cublas_error(
		cublasSgemm(handle,
			trans_A, trans_B,
			m, n, k,
			&alpha,
			A.data(), A.rows(),
			B.data(), B.rows(),
			&beta,
			C.data(), C.rows()),
		"Failed to perform matrix multiplication");
	
	check_cublas_error(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

	return C;
}


__global__ void cuda_mat_mult_BLAS1_kernel(const float* a, float val, float* res, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		res[col * rows + row] = a[col * rows + row] * val;
	}
}

tfm::Tensor cuda_mat_mult_BLAS1(const tfm::Tensor& A, float val) {

	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = A.cols();
	size_t rows = A.rows();

	const_cast<tfm::Tensor&>(A).move_to(tfm::Device(tfm::DeviceType::CUDA, 0));
	tfm::Tensor res(A.cols(), A.rows(), A.device());

	// Launch a kernel on the GPU with one thread for each element.
	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	cuda_mat_mult_BLAS1_kernel<<<gridSize, blockSize>>>(A.data(), val, res.data(), cols, rows);

	// Check for any errors launching the kernel
	check_cuda_error(cudaGetLastError(), "cuda_mat_mult_BLAS1_kernel launch failed");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
	
	return res;
}


tfm::Tensor cuda_mat_div_BLAS3(const tfm::Tensor& A, const tfm::Tensor& B) {
	// TODO: implement
	return tfm::Tensor();
}


tfm::Tensor cuda_mat_div_BLAS1(const tfm::Tensor& A, float val) {
	// TODO: implement
	return tfm::Tensor();
}


__global__ void cuda_normalize_matrix_kernel(float* data, float* weights, float* bias, float* allocatedMem, size_t cols, size_t rows) {
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
		mean[row] /= cols;
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
		stddev[row] = sqrtf(stddev[row] / cols);
	}
	__syncthreads();

	if (row < rows) {
		for (int i = col; i < cols; i += blockDim.x) {
			data[i * rows + row] = ((data[i * rows + row] - mean[row]) / stddev[row]) * weights[row] + bias[row];
		}
	}
}

void cuda_normalize_matrix(tfm::Tensor& matrix) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = matrix.cols();
	size_t rows = matrix.rows();

	matrix.move_to(tfm::Device(tfm::DeviceType::CUDA, 0));
	
	// Launch a kernel on the GPU with one thread for each element.
	dim3 blockSize(16, 16);
	dim3 gridSize((rows + blockSize.y - 1) / blockSize.y);

	float* mem = nullptr;
	check_cuda_error(cudaMalloc((void**)&mem, 2 * rows * sizeof(float)), "cudaMalloc failed");
	check_cuda_error(cudaMemset(mem, 0, 2 * rows * sizeof(float)), "cudaMemset failed");
	cuda_normalize_matrix_kernel<<<gridSize, blockSize>>>(matrix.data(), matrix.weights(), matrix.bias(), mem, cols, rows);

	// Check for any errors launching the kernel
	check_cuda_error(cudaGetLastError(), "cuda_normalize_matrix_kernel launch failed");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
	check_cuda_error(cudaFree(mem), "cudaFree failed");

	return;
}


__global__ void cuda_ReLU_kernel(float* data, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows && data[col * rows + row] < 0) {
		data[col * rows + row] = 0;
	}
}

void cuda_ReLU(tfm::Tensor& matrix) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = matrix.cols();
	size_t rows = matrix.rows();

	matrix.move_to(tfm::Device(tfm::DeviceType::CUDA, 0));
	
	// Launch a kernel on the GPU with one thread for each element.
	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	cuda_ReLU_kernel<<<gridSize, blockSize>>>(matrix.data(), cols, rows);

	// Check for any errors launching the kernel
	check_cuda_error(cudaGetLastError(), "cuda_ReLU_kernel launch failed");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

	return;
}


void cuda_ReLU_derivative(tfm::Tensor& matrix) {
	// TODO: implement

}


__global__ void cuda_softmax_kernel(float* data, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float max_val = -FLT_MAX;
	for (size_t row = 0; row < rows; row++) {
		max_val = fmaxf(max_val, data[col * rows + row]);
	}

	float sum_exp = 0.0f;
	for (size_t row = 0; row < rows; row++) {
		data[col * rows + row] = expf(data[col * rows + row] - max_val);
		sum_exp += data[col * rows + row];
	}

	for (size_t row = 0; row < rows; row++) {
		data[col * rows + row] /= sum_exp;
	}
}

void cuda_softmax(tfm::Tensor& matrix) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = matrix.cols();
	size_t rows = matrix.rows();

	matrix.move_to(tfm::Device(tfm::DeviceType::CUDA, 0));

	// Launch a kernel on the GPU with one thread for each element.
	int blockSize = 16 * 16;
	int numBlocks = (cols + blockSize - 1) / blockSize;

	cuda_softmax_kernel<<<numBlocks, blockSize>>>(matrix.data(), cols, rows);

	// Check for any errors launching the kernel
	check_cuda_error(cudaGetLastError(), "cuda_softmax_kernel launch failed");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

	return;
}


void cuda_sq(tfm::Tensor& matrix) {
	// TODO: implement

}


void cuda_sqrt(tfm::Tensor& matrix) {
	// TODO: implement

}


