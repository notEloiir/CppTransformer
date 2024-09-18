
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

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

tfm::Tensor cuda_mat_add(const tfm::Tensor& A, const tfm::Tensor& B) {
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

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	if (B.is_vector()) {
		cuda_mat_add_BLAS2_kernel<<<gridSize, blockSize>>>(A.data(), B.data(), C.data(), cols, rows);
	}
	else {
		cuda_mat_add_BLAS3_kernel<<<gridSize, blockSize>>>(A.data(), B.data(), C.data(), cols, rows);
	}

	check_cuda_error(cudaGetLastError(), "cuda_mat_add_BLAS3_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

	return C;
}


void cuda_mat_add_inplace(tfm::Tensor& A, const tfm::Tensor& B) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = A.cols() < B.cols() ? A.cols() : B.cols();
	size_t rows = A.rows() < B.rows() ? A.rows() : B.rows();

	if (B.is_vector()) {
		cols = A.cols();
	}

	tfm::Device device(tfm::DeviceType::CUDA, 0);
	const_cast<tfm::Tensor&>(A).move_to(device);
	const_cast<tfm::Tensor&>(B).move_to(device);

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	if (B.is_vector()) {
		cuda_mat_add_BLAS2_kernel<<<gridSize, blockSize>>>(A.data(), B.data(), A.data(), cols, rows);
	}
	else {
		cuda_mat_add_BLAS3_kernel<<<gridSize, blockSize>>>(A.data(), B.data(), A.data(), cols, rows);
	}

	check_cuda_error(cudaGetLastError(), "cuda_mat_add_BLAS3_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}


__global__ void cuda_mat_add_along_axis_kernel(const float* matrix, float* res, size_t cols, size_t rows, bool axis) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	size_t res_id = (axis == 0) ? row : col;
	if (col < cols && row < rows) {
		atomicAdd(&res[res_id], matrix[col * rows + row]);
	}
}


tfm::Tensor cuda_mat_add_along_axis(const tfm::Tensor& A, size_t axis) {
	if (axis > 1) {
		throw std::invalid_argument("axis > 1 not supported");
	}

	size_t cols = axis == 0 ? 1 : A.cols();
	size_t rows = axis == 0 ? A.rows() : 1;

	tfm::Tensor res(cols, rows, A.device());

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	cuda_mat_add_along_axis_kernel<<<gridSize, blockSize>>>(A.data(), res.data(), A.cols(), A.rows(), axis);

	check_cuda_error(cudaGetLastError(), "cuda_mat_add kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

	return res;
}


__global__ void cuda_mat_sub_BLAS2_kernel(const float* a, const float* b, float* c, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		c[col * rows + row] = a[col * rows + row] - b[row];
	}
}

__global__ void cuda_mat_sub_BLAS3_kernel(const float* a, const float* b, float* c, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		c[col * rows + row] = a[col * rows + row] - b[col * rows + row];
	}
}

tfm::Tensor cuda_mat_sub(const tfm::Tensor& A, const tfm::Tensor& B) {
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

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	if (B.is_vector()) {
		cuda_mat_sub_BLAS2_kernel<<<gridSize, blockSize>>>(A.data(), B.data(), C.data(), cols, rows);
	}
	else {
		cuda_mat_sub_BLAS3_kernel<<<gridSize, blockSize>>>(A.data(), B.data(), C.data(), cols, rows);
	}

	check_cuda_error(cudaGetLastError(), "cuda_mat_sub kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

	return C;
}


tfm::Tensor cuda_mat_mult(const tfm::Tensor& A, const tfm::Tensor& B, bool transpose_A, bool transpose_B) {
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


__global__ void cuda_mat_mult_elementwise_kernel(const float* a, const float* b, float* c, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		c[col * rows + row] = a[col * rows + row] * b[col * rows + row];
	}
}

tfm::Tensor cuda_mat_mult_elementwise(const tfm::Tensor& A, const tfm::Tensor& B) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = A.cols() < B.cols() ? A.cols() : B.cols();
	size_t rows = A.rows() < B.rows() ? A.rows() : B.rows();

	tfm::Device device(tfm::DeviceType::CUDA, 0);
	const_cast<tfm::Tensor&>(A).move_to(device);
	const_cast<tfm::Tensor&>(B).move_to(device);
	tfm::Tensor C(cols, rows, device);

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	cuda_mat_mult_elementwise_kernel<<<gridSize, blockSize>>>(A.data(), B.data(), C.data(), cols, rows);

	check_cuda_error(cudaGetLastError(), "cuda_mat_mult_elementwise_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

	return C;
}


__global__ void cuda_mat_mult_BLAS1_kernel(const float* a, float val, float* res, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		res[col * rows + row] = a[col * rows + row] * val;
	}
}


tfm::Tensor cuda_mat_mult(const tfm::Tensor& A, float val) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = A.cols();
	size_t rows = A.rows();

	const_cast<tfm::Tensor&>(A).move_to(tfm::Device(tfm::DeviceType::CUDA, 0));
	tfm::Tensor res(A.cols(), A.rows(), A.device());

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	cuda_mat_mult_BLAS1_kernel<<<gridSize, blockSize>>>(A.data(), val, res.data(), cols, rows);

	check_cuda_error(cudaGetLastError(), "cuda_mat_mult_BLAS1_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
	
	return res;
}


__global__ void cuda_mat_div_elementwise_kernel(const float* a, const float* b, float* c, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		c[col * rows + row] = a[col * rows + row] * b[col * rows + row];
	}
}

tfm::Tensor cuda_mat_div_elementwise(const tfm::Tensor& A, const tfm::Tensor& B) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = A.cols() < B.cols() ? A.cols() : B.cols();
	size_t rows = A.rows() < B.rows() ? A.rows() : B.rows();

	tfm::Device device(tfm::DeviceType::CUDA, 0);
	const_cast<tfm::Tensor&>(A).move_to(device);
	const_cast<tfm::Tensor&>(B).move_to(device);
	tfm::Tensor C(cols, rows, device);

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	cuda_mat_div_elementwise_kernel<<<gridSize, blockSize>>>(A.data(), B.data(), C.data(), cols, rows);

	check_cuda_error(cudaGetLastError(), "cuda_mat_div_elementwise_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

	return C;
}


__global__ void cuda_mat_div_BLAS1_kernel(const float* a, float val, float* res, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		res[col * rows + row] = a[col * rows + row] * val;
	}
}


tfm::Tensor cuda_mat_div(const tfm::Tensor& A, float val) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = A.cols();
	size_t rows = A.rows();

	const_cast<tfm::Tensor&>(A).move_to(tfm::Device(tfm::DeviceType::CUDA, 0));
	tfm::Tensor res(A.cols(), A.rows(), A.device());

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	cuda_mat_div_BLAS1_kernel<<<gridSize, blockSize>>>(A.data(), val, res.data(), cols, rows);

	check_cuda_error(cudaGetLastError(), "cuda_mat_div_BLAS1_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
	
	return res;
}


__global__ void cuda_sq_kernel(float* matrix, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		matrix[col * rows + row] = matrix[col * rows + row] * matrix[col * rows + row];
	}
}

void cuda_sq(tfm::Tensor& matrix) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	matrix.move_to(tfm::Device(tfm::DeviceType::CUDA, 0));

	dim3 blockSize(16, 16);
	dim3 gridSize((matrix.cols() + blockSize.x - 1) / blockSize.x, (matrix.rows() + blockSize.y - 1) / blockSize.y);

	cuda_sq_kernel<<<gridSize, blockSize>>>(matrix.data(), matrix.cols(), matrix.rows());

	check_cuda_error(cudaGetLastError(), "cuda_sq_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}


__global__ void cuda_sqrt_kernel(float* matrix, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		matrix[col * rows + row] = sqrtf(matrix[col * rows + row]);
	}
}

void cuda_sqrt(tfm::Tensor& matrix) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	matrix.move_to(tfm::Device(tfm::DeviceType::CUDA, 0));

	dim3 blockSize(16, 16);
	dim3 gridSize((matrix.cols() + blockSize.x - 1) / blockSize.x, (matrix.rows() + blockSize.y - 1) / blockSize.y);

	cuda_sqrt_kernel<<<gridSize, blockSize>>>(matrix.data(), matrix.cols(), matrix.rows());

	check_cuda_error(cudaGetLastError(), "cuda_sqrt_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}


