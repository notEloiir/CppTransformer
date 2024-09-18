
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <backend/cuda/cuda_tensor_operations.cuh>
#include <cuda_utils.h>


__device__ float atomicMaxFloat(float* addr, float value) {
	if (value >= 0.0f) {
		return __int_as_float(atomicMax((int*)addr, __float_as_int(value)));
	}
	else {
		return __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
	}
}


__global__ void cuda_normalize_matrix_kernel(float* data, float* weights, float* bias, float* allocated_mem, size_t cols, size_t rows) {
	float* mean = allocated_mem;
	float* stddev = allocated_mem + rows;

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < rows && col < cols) {
		atomicAdd(&mean[row], data[col * rows + row]);
	}
	__syncthreads();

	if (row < rows && col == 0) {
		mean[row] /= cols;
	}
	__syncthreads();

	if (row < rows && col < cols) {
		float diff = data[col * rows + row] - mean[row];
		atomicAdd(&stddev[row], diff * diff);
	}
	__syncthreads();

	if (row < rows && col == 0) {
		stddev[row] = sqrtf(stddev[row] / cols);
	}
	__syncthreads();

	if (row < rows && col < cols) {
		data[col * rows + row] = ((data[col * rows + row] - mean[row]) / stddev[row]) * weights[row] + bias[row];
	}
}

void cuda_normalize_matrix(tfm::Tensor& matrix) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = matrix.cols();
	size_t rows = matrix.rows();

	matrix.move_to(tfm::Device(tfm::DeviceType::CUDA, 0));

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	float* mem = nullptr;
	check_cuda_error(cudaMalloc((void**)&mem, 2 * rows * sizeof(float)), "cudaMalloc failed");
	check_cuda_error(cudaMemset(mem, 0, 2 * rows * sizeof(float)), "cudaMemset failed");
	cuda_normalize_matrix_kernel<<<gridSize, blockSize>>>(matrix.data(), matrix.weights(), matrix.bias(), mem, cols, rows);

	check_cuda_error(cudaGetLastError(), "cuda_normalize_matrix_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
	check_cuda_error(cudaFree(mem), "cudaFree failed");
}


__global__ void cuda_normalize_matrix_backward_kernel(
	float* input, float* input_weights, float* input_bias,
	float* grad, float* grad_weights, float* grad_bias,
	float* allocated_mem, size_t cols, size_t rows) {

	float* mean = allocated_mem;
	float* stddev = allocated_mem + rows;
	float* grad_mean = allocated_mem + 2 * rows;
	float* grad_var = allocated_mem + 3 * rows;

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < rows && col < cols) {
		atomicAdd(&mean[row], input[col * rows + row]);
	}
	__syncthreads();

	if (row < rows && col == 0) {
		mean[row] /= cols;
	}
	__syncthreads();

	if (row < rows && col < cols) {
		float diff = input[col * rows + row] - mean[row];
		atomicAdd(&stddev[row], diff * diff);
	}
	__syncthreads();

	if (row < rows && col == 0) {
		stddev[row] = sqrtf(stddev[row] / cols);

		grad_weights[row] = 0.0f;
		grad_bias[row] = 0.0f;
	}
	__syncthreads();

	if (row < rows && col < cols) {
		// Calculate gradients for gamma and beta
		atomicAdd(&grad_weights[row], (grad[col * rows + row] * (input[col * rows + row] - input_bias[row]) / input_weights[row]));
		atomicAdd(&grad_bias[row], grad[col * rows + row]);

		// Gradient wrt the normalized output
		float grad_norm = grad[col * rows + row] * input_weights[row];

		// Accumulate gradients wrt variance and mean
		atomicAdd(&grad_var[row], (grad_norm * (input[col * rows + row] - mean[row]) * -0.5f * powf((stddev[row] * stddev[row]) + FLT_MIN, -1.5f)));
		atomicAdd(&grad_mean[row], (grad_norm * -1.0f / stddev[row]));
	}
	__syncthreads();

	// Gradient wrt input
	if (row < rows && col < cols) {
		float grad_input_cell = (grad[col * rows + row] * input_weights[row]) / stddev[row];
		grad_input_cell += grad_var[row] * 2.0f * (input[col * rows + row] - mean[row]) / cols;
		grad_input_cell += grad_mean[row] / cols;

		grad[col * rows + row] = grad_input_cell;
	}

	if (row < rows && col == 0) {
		atomicAdd(&input_weights[row], -grad_weights[row]);
		atomicAdd(&input_bias[row], -grad_bias[row]);
	}
}

void cuda_normalize_matrix_backward(tfm::Tensor& grad, const tfm::Tensor& normalize_input) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = normalize_input.cols();
	size_t rows = normalize_input.rows();

	const_cast<tfm::Tensor &>(normalize_input).move_to(tfm::Device(tfm::DeviceType::CUDA, 0));
	grad.move_to(tfm::Device(tfm::DeviceType::CUDA, 0));

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
	
	float* mem = nullptr;
	check_cuda_error(cudaMalloc((void**)&mem, 4 * rows * sizeof(float)), "cudaMalloc failed");
	check_cuda_error(cudaMemset(mem, 0, 4 * rows * sizeof(float)), "cudaMemset failed");

	cuda_normalize_matrix_backward_kernel<<<gridSize, blockSize>>>(
		normalize_input.data(), normalize_input.weights(), normalize_input.bias(),
		grad.data(), grad.weights(), grad.bias(),
		mem, cols, rows);

	check_cuda_error(cudaGetLastError(), "cuda_normalize_matrix_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
	check_cuda_error(cudaFree(mem), "cudaFree failed");
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
	
	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	cuda_ReLU_kernel<<<gridSize, blockSize>>>(matrix.data(), cols, rows);

	check_cuda_error(cudaGetLastError(), "cuda_ReLU_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}


__global__ void cuda_ReLU_derivative_kernel(float* data, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows && data[col * rows + row] < 0) {
		data[col * rows + row] = 0;
	}
}

void cuda_ReLU_derivative(tfm::Tensor& matrix) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = matrix.cols();
	size_t rows = matrix.rows();

	matrix.move_to(tfm::Device(tfm::DeviceType::CUDA, 0));

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	cuda_ReLU_derivative_kernel<<<gridSize, blockSize>>>(matrix.data(), cols, rows);

	check_cuda_error(cudaGetLastError(), "cuda_ReLU_derivative_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}


__global__ void cuda_softmax_kernel(float* data, float* allocated_mem, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	float* max_val = allocated_mem;
	float* sum_exp = allocated_mem + cols;

	if (col < cols && row == 0) {
		max_val[col] = -FLT_MAX;
	}
	__syncthreads();

	if (col < cols && row < rows) {
		if (max_val[col] < data[col * rows + row]) {
			atomicMaxFloat(&max_val[col], data[col * rows + row]);
		}
	}
	__syncthreads();

	if (col < cols && row < rows) {
		data[col * rows + row] = expf(data[col * rows + row] - max_val[col]);
		atomicAdd(&sum_exp[col], data[col * rows + row]);
	}
	__syncthreads();

	if (col < cols && row < rows) {
		data[col * rows + row] /= sum_exp[col];
	}
}

void cuda_softmax(tfm::Tensor& matrix) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = matrix.cols();
	size_t rows = matrix.rows();

	matrix.move_to(tfm::Device(tfm::DeviceType::CUDA, 0));

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	float* mem = nullptr;
	check_cuda_error(cudaMalloc((void**)&mem, 2 * cols * sizeof(float)), "cudaMalloc failed");
	check_cuda_error(cudaMemset(mem, 0, 2 * cols * sizeof(float)), "cudaMemset failed");

	cuda_softmax_kernel<<<gridSize, blockSize>>>(matrix.data(), mem, cols, rows);

	check_cuda_error(cudaGetLastError(), "cuda_softmax_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
	check_cuda_error(cudaFree(mem), "cudaFree failed");
}


__global__ void cuda_softmax_backward_kernel(float* grad, float* softmax_output, float* allocated_mem, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	float* dot_product = allocated_mem;

	// Calculate the dot product of the softmax vector with the gradient vector
	if (col < cols && row < rows) {
		atomicAdd(&dot_product[col], softmax_output[col * rows + row] * grad[col * rows + row]);
	}
	__syncthreads();

	// Compute the gradient for each element in the column
	if (col < cols && row < rows) {
		// For each element in the column, apply the softmax derivative formula:
		// grad = softmax * (grad - (softmax * sum(grad)))
		grad[col * rows + row] = softmax_output[col * rows + row] * (grad[col * rows + row] - dot_product[col]);
	}
}

void cuda_softmax_backward(tfm::Tensor& grad, const tfm::Tensor& softmax_output) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	size_t cols = grad.cols();
	size_t rows = grad.rows();

	const_cast<tfm::Tensor&>(softmax_output).move_to(tfm::Device(tfm::DeviceType::CUDA, 0));
	grad.move_to(tfm::Device(tfm::DeviceType::CUDA, 0));

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	float* mem = nullptr;
	check_cuda_error(cudaMalloc((void**)&mem, 1 * cols * sizeof(float)), "cudaMalloc failed");
	check_cuda_error(cudaMemset(mem, 0, 1 * cols * sizeof(float)), "cudaMemset failed");

	cuda_softmax_backward_kernel<<<gridSize, blockSize>>>(grad.data(), softmax_output.data(), mem, cols, rows);

	check_cuda_error(cudaGetLastError(), "cuda_softmax_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
	check_cuda_error(cudaFree(mem), "cudaFree failed");
}
