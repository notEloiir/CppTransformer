#include <tensor/device.h>
#include <cuda_utils.h>


tfm::Device::Device(DeviceType type, int index) noexcept :
	type(type),
	id(index)
{}


const int tfm::Device::deviceCount = []() -> int {
    int count = 0;
    checkCudaError(cudaGetDeviceCount(&count), "cudaGetDeviceCount failed");
    return count;
   }();
