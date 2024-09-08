#include <tensor/device.h>
#include <cuda_utils.h>


tfm::Device::Device(DeviceType type, int index) noexcept :
	type_(type),
	id_(index)
{}


const int tfm::Device::device_count = []() -> int {
    int count = 0;

	#ifndef NO_CUDA
	check_cuda_error(cudaGetDeviceCount(&count), "cudaGetDeviceCount failed");
	#endif // !NO_CUDA

    return count;
   }();
