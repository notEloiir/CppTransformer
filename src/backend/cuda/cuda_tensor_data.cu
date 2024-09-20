
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <backend/cuda/cuda_tensor_data.cuh>
#include <cuda_utils.h>


__global__ void cuda_fill_kernel(float* dev_ptr, float val, size_t cols, size_t rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		dev_ptr[col * rows + row] = val;
	}
}


void cuda_fill(float* dev_ptr, float val, size_t cols, size_t rows) {
	check_cuda_error(cudaSetDevice(0), "cudaSetDevice failed");

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	cuda_fill_kernel<<<gridSize, blockSize>>>(dev_ptr, val, cols, rows);

	check_cuda_error(cudaGetLastError(), "cuda_fill_kernel launch failed");
	check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}
