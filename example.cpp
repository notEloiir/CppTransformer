/*
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <tensor/tensor.h>
#include <cuda_utils.h>


int main() {
	srand(time(NULL));

	tfm::Tensor t1(2, 3, tfm::Device(tfm::DeviceType::CPU));
	for (size_t col = 0; col < t1.cols(); col++) {
		for (size_t row = 0; row < t1.rows(); row++) {
			t1[col][row] = (1 + col * t1.rows() + row);
		}
	}

	tfm::Tensor t2 = t1.nonOwningCopy({ 1, 0 });

	// t1 * t2.T
	tfm::Tensor t3 = t1.multiply(t2, false, true);

	std::cout << t1 << std::endl;
	std::cout << t2 << std::endl;
	std::cout << t3 << std::endl;

	#ifndef NO_CUDA
	checkCudaError(cudaDeviceReset(), "cudaDeviceReset failed!");
	#endif // !NO_CUDA

	return 0;
}

*/