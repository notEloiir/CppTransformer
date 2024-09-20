
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <cuda_runtime.h>

#include <tensor/tensor.h>
#include <cuda_utils.h>
#include <compiler_flags.h>
#include <backend/cuda/cuda_tensor_data.cuh>


tfm::Tensor::Tensor() :
	cols_(0),
	rows_(0),
	data_(nullptr),
	data_cuda_(nullptr),
	data_2D_(nullptr),
	weights_(nullptr),
	weights_cuda_(nullptr),
	bias_(nullptr),
	bias_cuda_(nullptr),
	is_owning_(false),
	is_owning_cuda_(false),
	is_data_continuous_(false),
	device_(tfm::DeviceType::CPU) {}


tfm::Tensor::Tensor(size_t cols, size_t rows, Device device) :
	cols_(cols),
	rows_(rows),
	data_((float*)malloc(cols* rows * sizeof(float))),
	data_cuda_(nullptr),
	data_2D_((float**)malloc(cols * sizeof(float*))),
	weights_(nullptr),
	weights_cuda_(nullptr),
	bias_(nullptr),
	bias_cuda_(nullptr),
	is_owning_(true),
	is_owning_cuda_(false),
	is_data_continuous_(true),
	device_(device) {

	if (data_ == NULL || data_2D_ == NULL) {
		fprintf(stderr, "malloc failed");
		exit(1);
	}
	for (size_t col = 0; col < cols; col++) {
		data_2D_[col] = data_ + col * rows;
	}

	if (device.is_CUDA()) {
		allocate_cuda();
		is_owning_cuda_ = true;
	}
}


tfm::Tensor::Tensor(size_t cols, size_t rows, Device device, float* allocated_data) :
	cols_(cols),
	rows_(rows),
	data_(nullptr),
	data_cuda_(nullptr),
	data_2D_((float**)malloc(cols * sizeof(float*))),
	weights_(nullptr),
	weights_cuda_(nullptr),
	bias_(nullptr),
	bias_cuda_(nullptr),
	is_owning_(false),
	is_owning_cuda_(false),
	is_data_continuous_(true),
	device_(device) {

	if (data_2D_ == NULL) {
		fprintf(stderr, "malloc failed");
		exit(1);
	}
	for (size_t col = 0; col < cols; col++) {
		data_2D_[col] = data_ + col * rows;
	}

	if (device.is_CPU()) {
		data_ = allocated_data;
	}
	else {  // device.is_CUDA()
		data_cuda_ = allocated_data;
	}
}


tfm::Tensor::Tensor(const Tensor& other) :
	cols_(other.cols_),
	rows_(other.rows_),
	data_((float*)malloc(other.cols_* other.rows_ * sizeof(float))),
	data_cuda_(nullptr),
	data_2D_((float**)malloc(other.cols_ * sizeof(float*))),
	weights_(nullptr),
	weights_cuda_(nullptr),
	bias_(nullptr),
	bias_cuda_(nullptr),
	is_owning_(true),
	is_owning_cuda_(false),
	is_data_continuous_(true),
	device_(other.device_) {

	if (data_ == NULL || data_2D_ == NULL) {
		fprintf(stderr, "malloc failed");
		exit(1);
	}

	if (other.has_weights()) {
		weights_ = (float*)malloc(rows_ * sizeof(float));
		if (weights_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}
	if (other.has_bias()) {
		bias_ = (float*)malloc(rows_ * sizeof(float));
		if (bias_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}

	for (size_t col = 0; col < cols_; col++) {
		data_2D_[col] = data_ + col * rows_;
	}

	if (device_.is_CUDA()) {
		allocate_cuda();
		// copy GPU side
		check_cuda_error(cudaMemcpy(data_cuda_, other.data_cuda_, cols_ * rows_ * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy data_cuda_");
		if (other.has_weights()) {
			check_cuda_error(cudaMemcpy(weights_cuda_, other.weights_cuda_, rows_ * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy weights_cuda_");
		}
		if (other.has_bias()) {
			check_cuda_error(cudaMemcpy(bias_cuda_, other.bias_cuda_, rows_ * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy bias_cuda_");
		}
	}
	else {  // device_.is_CPU()
		// copy RAM side
		if (other.is_data_continuous_) {
			memcpy(data_, other.data_, cols_ * rows_ * sizeof(float));
		}
		else {
			for (size_t col = 0; col < cols_; col++) {
				memcpy((data_ + col * rows_), other.data_2D_[col], rows_ * sizeof(float));
			}
		}

		if (other.has_weights()) {
			assert(weights_ != nullptr);
			memcpy(weights_, other.weights_, rows_ * sizeof(float));
		}
		if (other.has_bias()) {
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
	data_cuda_ = nullptr;
	data_2D_ = (float**)malloc(other.cols_ * sizeof(float*));
	weights_ = nullptr;
	weights_cuda_ = nullptr;
	bias_ = nullptr;
	bias_cuda_ = nullptr;
	is_owning_ = true;
	is_owning_cuda_ = false;
	is_data_continuous_ = true;
	device_ = other.device_;

	if (data_ == NULL || data_2D_ == NULL) {
		fprintf(stderr, "malloc failed");
		exit(1);
	}

	if (other.has_weights()) {
		weights_ = (float*)malloc(rows_ * sizeof(float));
		if (weights_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}
	if (other.has_bias()) {
		bias_ = (float*)malloc(rows_ * sizeof(float));
		if (bias_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}
	
	for (size_t col = 0; col < cols_; col++) {
		data_2D_[col] = data_ + col * rows_;
	}

	if (device_.is_CUDA()) {
		allocate_cuda();
		// copy GPU side
		check_cuda_error(cudaMemcpy(data_cuda_, other.data_cuda_, cols_ * rows_ * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy data_cuda_");
		if (other.has_weights()) {
			check_cuda_error(cudaMemcpy(weights_cuda_, other.weights_cuda_, rows_ * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy weights_cuda_");
		}
		if (other.has_bias()) {
			check_cuda_error(cudaMemcpy(bias_cuda_, other.bias_cuda_, rows_ * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy bias_cuda_");
		}
	}
	else {  // device_.is_CPU()
		// copy RAM side
		if (other.is_data_continuous_) {
			memcpy(data_, other.data_, cols_ * rows_ * sizeof(float));
		}
		else {
			for (size_t col = 0; col < cols_; col++) {
				memcpy((data_ + col * rows_), other.data_2D_[col], rows_ * sizeof(float));
			}
		}

		if (other.has_weights()) {
			assert(weights_ != nullptr);
			memcpy(weights_, other.weights_, rows_ * sizeof(float));
		}
		if (other.has_bias()) {
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
	data_cuda_(other.data_cuda_),
	data_2D_(other.data_2D_),
	weights_(other.weights_),
	weights_cuda_(other.weights_cuda_),
	bias_(other.bias_),
	bias_cuda_(other.bias_cuda_),
	is_owning_(other.is_owning_),
	is_owning_cuda_(other.is_owning_cuda_),
	is_data_continuous_(other.is_data_continuous_),
	device_(other.device_) {

	other.data_2D_ = nullptr;
	other.is_owning_ = false;
	other.is_owning_cuda_ = false;
	other.cleanup();
}

tfm::Tensor& tfm::Tensor::operator=(tfm::Tensor&& other) noexcept {
	if (this == &other) {
		return *this;
	}

	cols_ = other.cols_;
	rows_ = other.rows_;
	data_ = other.data_;
	data_cuda_ = other.data_cuda_;
	data_2D_ = other.data_2D_;
	weights_ = other.weights_;
	weights_cuda_ = other.weights_cuda_;
	bias_ = other.bias_;
	bias_cuda_ = other.bias_cuda_;
	is_owning_ = other.is_owning_;
	is_owning_cuda_ = other.is_owning_cuda_;
	is_data_continuous_ = other.is_data_continuous_;
	device_ = other.device_;

	other.data_2D_ = nullptr;
	other.is_owning_ = false;
	other.is_owning_cuda_ = false;
	other.cleanup();

	return *this;
}


tfm::Tensor tfm::Tensor::non_owning_copy() const {
	tfm::Tensor v;

	v.cols_ = cols_;
	v.rows_ = rows_;
	v.data_ = data_;
	v.data_cuda_ = data_cuda_;
	v.data_2D_ = (float**)malloc(v.cols_ * sizeof(float*));
	v.weights_ = weights_;
	v.weights_cuda_ = weights_cuda_;
	v.bias_ = bias_;
	v.bias_cuda_ = bias_cuda_;
	v.is_owning_ = false;
	v.is_owning_cuda_ = false;
	v.is_data_continuous_ = is_data_continuous_;
	v.device_ = device_;

	if (v.data_2D_ == NULL) {
		fprintf(stderr, "malloc failed");
		exit(1);
	}

	for (size_t col = 0; col < cols_; col++) {
		v.data_2D_[col] = data_2D_[col];
	}

	return v;
}


tfm::Tensor tfm::Tensor::non_owning_copy(const std::vector<size_t>& col_ids) const {
	tfm::Tensor v = non_owning_copy();
	v.is_data_continuous_ = false;
	if (v.cols_ != col_ids.size()) {
		v.cols_ = col_ids.size();
		float** data_2D_new = (float**)realloc((void*)v.data_2D_, v.cols_ * sizeof(float*));
		if (data_2D_new == NULL) {
			fprintf(stderr, "realloc failed");
			exit(1);
		}
		v.data_2D_ = data_2D_new;
	}

	for (size_t col = 0; col < v.cols(); col++) {
		v.data_2D_[col] = data_2D_[col_ids[col]];
	}

	return v;
}


tfm::Tensor tfm::Tensor::non_owning_copy(size_t cols, size_t col_offset) const {
	tfm::Tensor v = non_owning_copy();

	if (col_offset + cols > cols_) {
		fprintf(stderr, "col_offset + cols can't be larger than column count of this matrix");
		exit(1);
	}

	v.cols_ = cols;
	v.data_ = data_ + rows_ * col_offset;
	if (is_owning_cuda_) {
		v.data_cuda_ = col_data(col_offset);
	}

	for (size_t col = 0; col < cols; col++) {
		data_2D_[col] = data_2D_[col_offset + col];
	}

	return v;
}


void tfm::Tensor::reset() {
	fill(0.0f);
#ifdef SAVE_VRAM
	move_to(tfm::Device(tfm::DeviceType::CPU));
#endif // SAVE_VRAM
}


void tfm::Tensor::init_weights() {
	if (has_weights) return;

	if (weights_ == nullptr) {
		weights_ = (float*)malloc(rows_ * sizeof(float));
		if (weights_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}

	std::fill(weights_, weights_ + rows_, 1.0f);

	if (device_.is_CUDA()) {
		check_cuda_error(cudaMalloc((void**)&weights_cuda_, rows_ * sizeof(float)), "Failed to allocate device memory");
		cuda_fill(weights_cuda_, 1.0f, 1, rows_);
	}
}


void tfm::Tensor::init_bias() {
	if (has_bias) return;

	if (bias_ == nullptr) {
		bias_ = (float*)malloc(rows_ * sizeof(float));
		if (bias_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}

	std::fill(bias_, bias_ + rows_, 0.0f);

	if (device_.is_CUDA()) {
		check_cuda_error(cudaMalloc((void**)&bias_cuda_, rows_ * sizeof(float)), "Failed to allocate device memory");
		cuda_fill(bias_cuda_, 1.0f, 1, rows_);
	}
}


void tfm::Tensor::move_to(Device new_device){
	if (device_ == new_device) {
		return;
	}

	if (device_.is_CPU() && new_device.is_CUDA()) {
		device_ = new_device;
		allocate_cuda();

		if (is_data_continuous_) {
			check_cuda_error(cudaMemcpy((void*)data_cuda_, (const void*)data_, cols_ * rows_ * sizeof(float), cudaMemcpyHostToDevice), "Copy to device failed");
		}
		else {
			for (size_t col = 0; col < cols_; col++) {
				check_cuda_error(cudaMemcpy((void*)(data_cuda_ + col * rows_), (const void*)data_2D_[col], rows_ * sizeof(float), cudaMemcpyHostToDevice), "Copy to device failed");
			}
		}

		if (has_weights()) {
			check_cuda_error(cudaMemcpy((void*)weights_cuda_, (const void*)weights_, rows_ * sizeof(float), cudaMemcpyHostToDevice), "Copy to device failed");
		}
		if (has_bias()) {
			check_cuda_error(cudaMemcpy((void*)bias_cuda_, (const void*)bias_, rows_ * sizeof(float), cudaMemcpyHostToDevice), "Copy to device failed");
		}
	}
	else if (device_.is_CUDA() && new_device.is_CPU()) {
		set_device();
		if (!is_owning_) {
			allocate();
		}

		check_cuda_error(cudaMemcpy((void*)data_, (const void*)data_cuda_, cols_ * rows_ * sizeof(float), cudaMemcpyDeviceToHost), "Copy to host failed");
		if (has_weights()) {
			check_cuda_error(cudaMemcpy((void*)weights_, (const void*)weights_cuda_, rows_ * sizeof(float), cudaMemcpyDeviceToHost), "Copy to host failed");
		}
		if (has_bias()) {
			check_cuda_error(cudaMemcpy((void*)bias_, (const void*)bias_cuda_, rows_ * sizeof(float), cudaMemcpyDeviceToHost), "Copy to host failed");
		}

		deallocate_cuda();
		device_ = new_device;
		is_owning_ = true;
		is_data_continuous_ = true;
	}
	else {  // CUDA to CUDA
		float* prev_data_cuda = data_cuda_;
		float* prev_weights_cuda = weights_cuda_;
		float* prev_bias_cuda = bias_cuda_;
		bool prev_is_owning_cuda = is_owning_cuda_;
		tfm::Device prev_device = device_;

		device_ = new_device;
		allocate_cuda();

		check_cuda_error(cudaMemcpyPeer((void*)data_cuda_, device_.index(), (const void*)prev_data_cuda, prev_device.index(), cols_ * rows_ * sizeof(float)), "Copy to device failed");
		if (has_weights()) {
			check_cuda_error(cudaMemcpyPeer((void*)weights_cuda_, device_.index(), (const void*)prev_weights_cuda, prev_device.index(), rows_ * sizeof(float)), "Copy to device failed");
		}
		if (has_bias()) {
			check_cuda_error(cudaMemcpyPeer((void*)bias_cuda_, device_.index(), (const void*)prev_bias_cuda, prev_device.index(), rows_ * sizeof(float)), "Copy to device failed");
		}

		// switch to old members to deallocate previous memory
		float* new_data_cuda = data_cuda_;
		float* new_weights_cuda = weights_cuda_;
		float* new_bias_cuda = bias_cuda_;
		data_cuda_ = prev_data_cuda;
		weights_cuda_ = prev_weights_cuda;
		bias_cuda_ = prev_bias_cuda;
		device_ = prev_device;
		is_owning_cuda_ = prev_is_owning_cuda;

		deallocate_cuda();

		data_cuda_ = new_data_cuda;
		weights_cuda_ = new_weights_cuda;
		bias_cuda_ = new_bias_cuda;
		device_ = new_device;
		is_owning_cuda_ = true;
	}
}


void tfm::Tensor::copy_col(const tfm::Tensor& src, size_t from_col, size_t to_col) {
	if (from_col >= src.cols() || to_col >= cols()) {
		throw std::invalid_argument("from_col or to_col out of range");
	}
	if (rows() != src.rows()) {
		throw std::invalid_argument("size doesn't match");
	}

	if (device().is_CPU()) {
		memcpy(col_data(to_col), src.col_data(from_col), rows() * sizeof(float));
	}
	else {
		check_cuda_error(cudaMemcpy((void*)(data() + to_col * rows()), (const void*)(src.data() + from_col * rows()), rows() * sizeof(float), cudaMemcpyDeviceToDevice), "Copy failed");
	}
}


int tfm::Tensor::save_to_path(const std::string& path) const {
	std::ofstream file;
	file.open(path, std::ios::out | std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "file open failed");
		exit(1);
	}

	Device orig_device = device_;
	const_cast<tfm::Tensor*>(this)->move_to(Device(tfm::DeviceType::CPU));

	for (size_t col = 0; col < cols(); col++) {
		for (size_t row = 0; row < rows(); row++) {
			file.write(reinterpret_cast<const char*>(&data_2D_[col][row]), sizeof(float));
		}
	}

	if (has_weights() && has_bias()) {
		for (size_t row = 0; row < rows(); row++) {
			file.write(reinterpret_cast<const char*>(&weights_[row]), sizeof(float));
		}
		for (size_t row = 0; row < rows(); row++) {
			file.write(reinterpret_cast<const char*>(&bias_[row]), sizeof(float));
		}
	}

	file.close();
	const_cast<tfm::Tensor*>(this)->move_to(orig_device);
	return 0;
}


int tfm::Tensor::load_from_path(const std::string& path, bool load_weights_and_bias) {
	std::ifstream file;
	file.open(path, std::ios::in | std::ios::binary);
	if (!file.good()) {
		return 1;
	}
	if (!file.is_open()) {
		fprintf(stderr, "file open failed");
		exit(1);
	}

	Device orig_device = device_;
	move_to(Device(tfm::DeviceType::CPU));

	for (size_t col = 0; col < cols(); col++) {
		for (size_t row = 0; row < rows(); row++) {
			file.read(reinterpret_cast<char*>(&data_2D_[col][row]), sizeof(float));
		}
	}

	if (load_weights_and_bias) {
		if (!has_weights()) {
			weights_ = (float*)malloc(rows_ * sizeof(float));
			if (weights_ == NULL) {
				fprintf(stderr, "malloc failed");
				exit(1);
			}
		}
		if (!has_bias()) {
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
	move_to(orig_device);
	return 0;
}


void tfm::Tensor::allocate() {
	is_owning_ = true;
	is_data_continuous_ = true;

	data_ = (float*)malloc(cols_ * rows_ * sizeof(float));

	if (data_ == NULL) {
		fprintf(stderr, "malloc failed");
		exit(1);
	}

	if (has_weights()) {
		weights_ = (float*)malloc(rows_ * sizeof(float));
		if (weights_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}
	if (has_bias()) {
		bias_ = (float*)malloc(rows_ * sizeof(float));
		if (bias_ == NULL) {
			fprintf(stderr, "malloc failed");
			exit(1);
		}
	}

	for (size_t col = 0; col < cols_; col++) {
		data_2D_[col] = data_ + col * rows_;
	}
}


void tfm::Tensor::deallocate() {
	if (is_owning_) {
		std::free(data_);
		std::free(weights_);
		std::free(bias_);
	}
	is_owning_ = false;
}


void tfm::Tensor::set_device() {
	check_cuda_error(cudaSetDevice(device_.index()), "Failed to set device");
}


void tfm::Tensor::deallocate_cuda() {
	if (is_owning_cuda_) {
		set_device();
		cudaFree(data_cuda_);
		cudaFree(weights_cuda_);
		cudaFree(bias_cuda_);
	}
	is_owning_cuda_ = false;
}


void tfm::Tensor::allocate_cuda() {
	is_owning_cuda_ = true;
	set_device();

	check_cuda_error(cudaMalloc((void**)&data_cuda_, cols_ * rows_ * sizeof(float)), "Failed to allocate device memory");
	
	if (has_weights()) {
		check_cuda_error(cudaMalloc((void**)&weights_cuda_, rows_ * sizeof(float)), "Failed to allocate device memory");
	}
	if (has_bias()) {
		check_cuda_error(cudaMalloc((void**)&bias_cuda_, rows_ * sizeof(float)), "Failed to allocate device memory");
	}
}


void tfm::Tensor::cleanup() {
	deallocate();
	deallocate_cuda();
	std::free(data_2D_);

	cols_ = 0;
	rows_ = 0;
	data_ = nullptr;
	data_cuda_ = nullptr;
	data_2D_ = nullptr;
	weights_ = nullptr;
	weights_cuda_ = nullptr;
	bias_ = nullptr;
	bias_cuda_ = nullptr;
	is_owning_ = false;
	is_owning_cuda_ = false;
	is_data_continuous_ = false;
	device_ = tfm::DeviceType::CPU;
}
