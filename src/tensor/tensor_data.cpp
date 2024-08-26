
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <cuda_runtime.h>

#include <tensor/tensor.h>
#include <cuda_utils.h>


tfm::Tensor::Tensor() :
	cols_(0),
	rows_(0),
	data_(nullptr),
	dataCuda_(nullptr),
	data2D_(nullptr),
	weights_(nullptr),
	weightsCuda_(nullptr),
	bias_(nullptr),
	biasCuda_(nullptr),
	isOwning_(false),
	isOwningCuda_(false),
	isDataContinuous_(false),
	device_(tfm::DeviceType::CPU) {}


tfm::Tensor::Tensor(size_t cols, size_t rows, Device device) :
	cols_(cols),
	rows_(rows),
	data_((float*)malloc(cols* rows * sizeof(float))),
	dataCuda_(nullptr),
	data2D_((float**)malloc(cols * sizeof(float*))),
	weights_(nullptr),
	weightsCuda_(nullptr),
	bias_(nullptr),
	biasCuda_(nullptr),
	isOwning_(true),
	isOwningCuda_(false),
	isDataContinuous_(true),
	device_(device) {

	if (data_ == NULL || data2D_ == NULL) {
		fprintf(stderr, "malloc failed");
		exit(1);
	}
	for (size_t col = 0; col < cols; col++) {
		data2D_[col] = data_ + col * rows;
	}

	if (device.isCUDA()) {
		allocateCuda();
		isOwningCuda_ = true;
	}
}


tfm::Tensor::Tensor(const Tensor& other) :
	cols_(other.cols_),
	rows_(other.rows_),
	data_((float*)malloc(other.cols_* other.rows_ * sizeof(float))),
	dataCuda_(nullptr),
	data2D_((float**)malloc(other.cols_ * sizeof(float*))),
	weights_(nullptr),
	weightsCuda_(nullptr),
	bias_(nullptr),
	biasCuda_(nullptr),
	isOwning_(true),
	isOwningCuda_(false),
	isDataContinuous_(true),
	device_(other.device_) {

	if (data_ == NULL || data2D_ == NULL) {
		fprintf(stderr, "malloc failed");
		exit(1);
	}

	if (other.hasWeights()) {
		weights_ = (float*)malloc(rows_ * sizeof(float));
		if (weights_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}
	if (other.hasBias()) {
		bias_ = (float*)malloc(rows_ * sizeof(float));
		if (bias_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}

	for (size_t col = 0; col < cols_; col++) {
		data2D_[col] = data_ + col * rows_;
	}

	if (device_.isCUDA()) {
		allocateCuda();
		// copy GPU side
		checkCudaError(cudaMemcpy(dataCuda_, other.dataCuda_, cols_ * rows_ * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy dataCuda_");
		if (other.hasWeights()) {
			checkCudaError(cudaMemcpy(weightsCuda_, other.weightsCuda_, rows_ * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy weightsCuda_");
		}
		if (other.hasBias()) {
			checkCudaError(cudaMemcpy(biasCuda_, other.biasCuda_, rows_ * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy biasCuda_");
		}
	}
	else {  // device_.isCPU()
		// copy RAM side
		if (other.isDataContinuous_) {
			memcpy(data_, other.data_, cols_ * rows_ * sizeof(float));
		}
		else {
			for (size_t col = 0; col < cols_; col++) {
				memcpy((data_ + col * rows_), other.data2D_[col], rows_ * sizeof(float));
			}
		}

		if (other.hasWeights()) {
			assert(weights_ != nullptr);
			memcpy(weights_, other.weights_, rows_ * sizeof(float));
		}
		if (other.hasBias()) {
			assert(bias_ != nullptr);
			memcpy(bias_, other.bias_, rows_ * sizeof(float));
		}
	}
}


tfm::Tensor& tfm::Tensor::operator=(const Tensor& other) {
	if (this == &other) {
		return *this;
	}

	cleanup();

	cols_ = other.cols_;
	rows_ = other.rows_;
	data_ = (float*)malloc(other.cols_ * other.rows_ * sizeof(float));
	dataCuda_ = nullptr;
	data2D_ = (float**)malloc(other.cols_ * sizeof(float*));
	weights_ = nullptr;
	weightsCuda_ = nullptr;
	bias_ = nullptr;
	biasCuda_ = nullptr;
	isOwning_ = true;
	isOwningCuda_ = false;
	isDataContinuous_ = true;
	device_ = other.device_;

	if (data_ == NULL || data2D_ == NULL) {
		fprintf(stderr, "malloc failed");
		exit(1);
	}

	if (other.hasWeights()) {
		weights_ = (float*)malloc(rows_ * sizeof(float));
		if (weights_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}
	if (other.hasBias()) {
		bias_ = (float*)malloc(rows_ * sizeof(float));
		if (bias_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}
	
	for (size_t col = 0; col < cols_; col++) {
		data2D_[col] = data_ + col * rows_;
	}

	if (device_.isCUDA()) {
		allocateCuda();
		// copy GPU side
		checkCudaError(cudaMemcpy(dataCuda_, other.dataCuda_, cols_ * rows_ * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy dataCuda_");
		if (other.hasWeights()) {
			checkCudaError(cudaMemcpy(weightsCuda_, other.weightsCuda_, rows_ * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy weightsCuda_");
		}
		if (other.hasBias()) {
			checkCudaError(cudaMemcpy(biasCuda_, other.biasCuda_, rows_ * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy biasCuda_");
		}
	}
	else {  // device_.isCPU()
		// copy RAM side
		if (other.isDataContinuous_) {
			memcpy(data_, other.data_, cols_ * rows_ * sizeof(float));
		}
		else {
			for (size_t col = 0; col < cols_; col++) {
				memcpy((data_ + col * rows_), other.data2D_[col], rows_ * sizeof(float));
			}
		}

		if (other.hasWeights()) {
			assert(weights_ != nullptr);
			memcpy(weights_, other.weights_, rows_ * sizeof(float));
		}
		if (other.hasBias()) {
			assert(bias_ != nullptr);
			memcpy(bias_, other.bias_, rows_ * sizeof(float));
		}
	}

	return *this;
}


tfm::Tensor::Tensor(tfm::Tensor&& other) noexcept :
	cols_(other.cols_),
	rows_(other.rows_),
	data_(other.data_),
	dataCuda_(other.dataCuda_),
	data2D_(other.data2D_),
	weights_(other.weights_),
	weightsCuda_(other.weightsCuda_),
	bias_(other.bias_),
	biasCuda_(other.biasCuda_),
	isOwning_(other.isOwning_),
	isOwningCuda_(other.isOwningCuda_),
	isDataContinuous_(other.isDataContinuous_),
	device_(other.device_) {

	other.data2D_ = nullptr;
	other.isOwning_ = false;
	other.isOwningCuda_ = false;
	other.cleanup();
}

tfm::Tensor& tfm::Tensor::operator=(tfm::Tensor&& other) noexcept {
	if (this == &other) {
		return *this;
	}

	cols_ = other.cols_;
	rows_ = other.rows_;
	data_ = other.data_;
	dataCuda_ = other.dataCuda_;
	data2D_ = other.data2D_;
	weights_ = other.weights_;
	weightsCuda_ = other.weightsCuda_;
	bias_ = other.bias_;
	biasCuda_ = other.biasCuda_;
	isOwning_ = other.isOwning_;
	isOwningCuda_ = other.isOwningCuda_;
	isDataContinuous_ = other.isDataContinuous_;
	device_ = other.device_;

	other.data2D_ = nullptr;
	other.isOwning_ = false;
	other.isOwningCuda_ = false;
	other.cleanup();

	return *this;
}


tfm::Tensor tfm::Tensor::nonOwningCopy() const {
	tfm::Tensor v;

	v.cols_ = cols_;
	v.rows_ = rows_;
	v.data_ = data_;
	v.dataCuda_ = dataCuda_;
	v.data2D_ = (float**)malloc(v.cols_ * sizeof(float*));
	v.weights_ = weights_;
	v.weightsCuda_ = weightsCuda_;
	v.bias_ = bias_;
	v.biasCuda_ = biasCuda_;
	v.isOwning_ = false;
	v.isOwningCuda_ = false;
	v.isDataContinuous_ = isDataContinuous_;
	v.device_ = device_;

	if (v.data2D_ == NULL) {
		fprintf(stderr, "malloc failed");
		exit(1);
	}

	for (size_t col = 0; col < cols_; col++) {
		v.data2D_[col] = data2D_[col];
	}

	return v;
}


tfm::Tensor tfm::Tensor::nonOwningCopy(const std::vector<size_t>& colIds) const {
	tfm::Tensor v = nonOwningCopy();
	v.isDataContinuous_ = false;
	if (v.cols_ != colIds.size()) {
		v.cols_ = colIds.size();
		float** data2Dnew = (float**)realloc((void*)v.data2D_, v.cols_ * sizeof(float*));
		if (data2Dnew == NULL) {
			fprintf(stderr, "realloc failed");
			exit(1);
		}
		v.data2D_ = data2Dnew;
	}

	for (size_t col = 0; col < v.cols(); col++) {
		v.data2D_[col] = data2D_[colIds[col]];
	}

	return v;
}


tfm::Tensor tfm::Tensor::nonOwningCopy(size_t cols, size_t colOffset) const {
	tfm::Tensor v = nonOwningCopy();

	if (colOffset + cols > cols_) {
		fprintf(stderr, "colOffset + cols can't be larger than column count of this matrix");
		exit(1);
	}

	v.cols_ = cols;
	v.data_ = data_ + rows_ * colOffset;
	if (isOwningCuda_) {
		v.dataCuda_ = colData(colOffset);
	}

	for (size_t col = 0; col < cols; col++) {
		data2D_[col] = data2D_[colOffset + col];
	}

	return v;
}


void tfm::Tensor::initWeights() {
	Device origDevice = device_;
	moveTo(Device(tfm::DeviceType::CPU));

	if (weights_ == nullptr) {
		weights_ = (float*)malloc(rows_ * sizeof(float));
		if (weights_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}

	std::fill(weights_, weights_ + rows_, 1.0f);

	moveTo(origDevice);
}


void tfm::Tensor::initBias() {
	Device origDevice = device_;
	moveTo(Device(tfm::DeviceType::CPU));

	if (bias_ == nullptr) {
		bias_ = (float*)malloc(rows_ * sizeof(float));
		if (bias_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}

	std::fill(bias_, bias_ + rows_, 0.0f);

	moveTo(origDevice);
}


void tfm::Tensor::moveTo(Device newDevice){
	if (device_ == newDevice) {
		return;
	}

	if (device_.isCPU() && newDevice.isCUDA()) {
		device_ = newDevice;
		allocateCuda();

		if (isDataContinuous_) {
			checkCudaError(cudaMemcpy((void*)dataCuda_, (const void*)data_, cols_ * rows_ * sizeof(float), cudaMemcpyHostToDevice), "Copy to device failed");
		}
		else {
			for (size_t col = 0; col < cols_; col++) {
				checkCudaError(cudaMemcpy((void*)(dataCuda_ + col * rows_), (const void*)data2D_[col], rows_ * sizeof(float), cudaMemcpyHostToDevice), "Copy to device failed");
			}
		}

		if (hasWeights()) {
			checkCudaError(cudaMemcpy((void*)weightsCuda_, (const void*)weights_, rows_ * sizeof(float), cudaMemcpyHostToDevice), "Copy to device failed");
		}
		if (hasBias()) {
			checkCudaError(cudaMemcpy((void*)biasCuda_, (const void*)bias_, rows_ * sizeof(float), cudaMemcpyHostToDevice), "Copy to device failed");
		}
	}
	else if (device_.isCUDA() && newDevice.isCPU()) {
		setDevice();
		if (!isOwning_) {
			allocate();
		}

		checkCudaError(cudaMemcpy((void*)data_, (const void*)dataCuda_, cols_ * rows_ * sizeof(float), cudaMemcpyDeviceToHost), "Copy to host failed");
		if (hasWeights()) {
			checkCudaError(cudaMemcpy((void*)weights_, (const void*)weightsCuda_, rows_ * sizeof(float), cudaMemcpyDeviceToHost), "Copy to host failed");
		}
		if (hasBias()) {
			checkCudaError(cudaMemcpy((void*)bias_, (const void*)biasCuda_, rows_ * sizeof(float), cudaMemcpyDeviceToHost), "Copy to host failed");
		}

		deallocateCuda();
		device_ = newDevice;
		isOwning_ = true;
		isDataContinuous_ = true;
	}
	else {  // CUDA to CUDA
		float* prevDataCUDA = dataCuda_;
		float* prevWeightsCUDA = weightsCuda_;
		float* prevBiasCUDA = biasCuda_;
		bool prevIsOwningCUDA = isOwningCuda_;
		tfm::Device prevDevice = device_;

		device_ = newDevice;
		allocateCuda();

		checkCudaError(cudaMemcpyPeer((void*)dataCuda_, device_.index(), (const void*)prevDataCUDA, prevDevice.index(), cols_ * rows_ * sizeof(float)), "Copy to device failed");
		if (hasWeights()) {
			checkCudaError(cudaMemcpyPeer((void*)weightsCuda_, device_.index(), (const void*)prevWeightsCUDA, prevDevice.index(), rows_ * sizeof(float)), "Copy to device failed");
		}
		if (hasBias()) {
			checkCudaError(cudaMemcpyPeer((void*)biasCuda_, device_.index(), (const void*)prevBiasCUDA, prevDevice.index(), rows_ * sizeof(float)), "Copy to device failed");
		}

		// switch to old members to deallocate previous memory
		float* newDataCUDA = dataCuda_;
		float* newWeightsCUDA = weightsCuda_;
		float* newBiasCUDA = biasCuda_;
		dataCuda_ = prevDataCUDA;
		weightsCuda_ = prevWeightsCUDA;
		biasCuda_ = prevBiasCUDA;
		device_ = prevDevice;
		isOwningCuda_ = prevIsOwningCUDA;

		deallocateCuda();

		dataCuda_ = newDataCUDA;
		weightsCuda_ = newWeightsCUDA;
		biasCuda_ = newBiasCUDA;
		device_ = newDevice;
		isOwningCuda_ = true;
	}
}


int tfm::Tensor::saveToPath(const std::string& path) const {
	std::ofstream file;
	file.open(path, std::ios::out | std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "file open failed");
		exit(1);
	}

	Device origDevice = device_;
	const_cast<tfm::Tensor*>(this)->moveTo(Device(tfm::DeviceType::CPU));

	for (size_t col = 0; col < cols(); col++) {
		for (size_t row = 0; row < rows(); row++) {
			file.write(reinterpret_cast<const char*>(&data2D_[col][row]), sizeof(float));
		}
	}

	if (hasWeights() && hasBias()) {
		for (size_t row = 0; row < rows(); row++) {
			file.write(reinterpret_cast<const char*>(&weights_[row]), sizeof(float));
		}
		for (size_t row = 0; row < rows(); row++) {
			file.write(reinterpret_cast<const char*>(&bias_[row]), sizeof(float));
		}
	}

	file.close();
	const_cast<tfm::Tensor*>(this)->moveTo(origDevice);
	return 0;
}


int tfm::Tensor::loadFromPath(const std::string& path, bool loadWeightsAndBias) {
	std::ifstream file;
	file.open(path, std::ios::in | std::ios::binary);
	if (!file.good()) {
		return 1;
	}
	if (!file.is_open()) {
		fprintf(stderr, "file open failed");
		exit(1);
	}

	Device origDevice = device_;
	moveTo(Device(tfm::DeviceType::CPU));

	for (size_t col = 0; col < cols(); col++) {
		for (size_t row = 0; row < rows(); row++) {
			file.read(reinterpret_cast<char*>(&data2D_[col][row]), sizeof(float));
		}
	}

	if (loadWeightsAndBias) {
		if (!hasWeights()) {
			weights_ = (float*)malloc(rows_ * sizeof(float));
			if (weights_ == NULL) {
				fprintf(stderr, "malloc failed");
				exit(1);
			}
		}
		if (!hasBias()) {
			bias_ = (float*)malloc(rows_ * sizeof(float));
			if (bias_ == NULL) {
				fprintf(stderr, "malloc failed");
				exit(1);
			}
		}

		for (size_t row = 0; row < rows(); row++) {
			file.read(reinterpret_cast<char*>(&weights_[row]), sizeof(float));
		}
		for (size_t row = 0; row < rows(); row++) {
			file.read(reinterpret_cast<char*>(&bias_[row]), sizeof(float));
		}
	}

	file.close();
	moveTo(origDevice);
	return 0;
}


void tfm::Tensor::allocate() {
	isOwning_ = true;
	isDataContinuous_ = true;

	data_ = (float*)malloc(cols_ * rows_ * sizeof(float));

	if (data_ == NULL) {
		fprintf(stderr, "malloc failed");
		exit(1);
	}

	if (hasWeights()) {
		weights_ = (float*)malloc(rows_ * sizeof(float));
		if (weights_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}
	if (hasBias()) {
		bias_ = (float*)malloc(rows_ * sizeof(float));
		if (bias_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}

	for (size_t col = 0; col < cols_; col++) {
		data2D_[col] = data_ + col * rows_;
	}
}


void tfm::Tensor::deallocate() {
	if (isOwning_) {
		std::free(data_);
		std::free(weights_);
		std::free(bias_);
	}
	isOwning_ = false;
}


void tfm::Tensor::setDevice() {
	checkCudaError(cudaSetDevice(device_.index()), "Failed to set device");
}


void tfm::Tensor::deallocateCuda() {
	if (isOwningCuda_) {
		setDevice();
		cudaFree(dataCuda_);
		cudaFree(weightsCuda_);
		cudaFree(biasCuda_);
	}
	isOwningCuda_ = false;
}


void tfm::Tensor::allocateCuda() {
	isOwningCuda_ = true;
	setDevice();

	checkCudaError(cudaMalloc((void**)&dataCuda_, cols_ * rows_ * sizeof(float)), "Failed to allocate device memory");
	
	if (hasWeights()) {
		checkCudaError(cudaMalloc((void**)&weightsCuda_, rows_ * sizeof(float)), "Failed to allocate device memory");
	}
	if (hasBias()) {
		checkCudaError(cudaMalloc((void**)&biasCuda_, rows_ * sizeof(float)), "Failed to allocate device memory");
	}
}


void tfm::Tensor::cleanup() {
	deallocate();
	deallocateCuda();
	std::free(data2D_);

	cols_ = 0;
	rows_ = 0;
	data_ = nullptr;
	dataCuda_ = nullptr;
	data2D_ = nullptr;
	weights_ = nullptr;
	weightsCuda_ = nullptr;
	bias_ = nullptr;
	biasCuda_ = nullptr;
	isOwning_ = false;
	isOwningCuda_ = false;
	isDataContinuous_ = false;
	device_ = tfm::DeviceType::CPU;
}
