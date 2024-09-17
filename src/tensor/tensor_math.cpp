
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <fstream>

#include <backend/cpu/cpu_matrix_math.h>
#include <backend/cpu/cpu_tensor_operations.h>
#include <backend/cuda/cuda_matrix_math.cuh>
#include <backend/cuda/cuda_tensor_data.cuh>
#include <backend/cuda/cuda_tensor_operations.cuh>
#include <tensor/tensor.h>


void tfm::Tensor::fill(float val) {
	tfm::Device dev = device_;
	move_to(tfm::Device(tfm::DeviceType::CPU));

	if (is_data_continuous_) {
		std::fill(data_, data_ + cols_ * rows_, val);
	}
	else {
		for (size_t col = 0; col < cols(); col++) {
			std::fill(data_2D_[col], data_2D_[col] + rows_, val);
		}
	}

	move_to(dev);
}


void tfm::Tensor::random() {
	tfm::Device dev = device_;
	move_to(tfm::Device(tfm::DeviceType::CPU));

	for (size_t col = 0; col < cols(); col++) {
		for (size_t row = 0; row < rows(); row++) {
			data_2D_[col][row] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2));
		}
	}
	
	move_to(dev);
}


void tfm::Tensor::normalize() {
	if (tfm::Device::device_count > 0) {
		cuda_normalize_matrix(*this);
	}
	else {
		cpu_normalize_matrix(*this);
	}
}


void tfm::Tensor::normalize_backward(const tfm::Tensor& normalize_input) {
	if (tfm::Device::device_count > 0) {
		cuda_normalize_matrix_backward(*this, normalize_input);
	}
	else {
		cpu_normalize_matrix_backward(*this, normalize_input);
	}
}


void tfm::Tensor::ReLU() {
	if (tfm::Device::device_count > 0) {
		cuda_ReLU(*this);
	}
	else {
		cpu_ReLU(*this);
	}
}


void tfm::Tensor::ReLU_derivative() {
	if (tfm::Device::device_count > 0) {
		cuda_ReLU_derivative(*this);
	}
	else {
		cpu_ReLU_derivative(*this);
	}
}


tfm::Tensor tfm::Tensor::multiply_elementwise_ReLU_derivative(const tfm::Tensor& other) const {
	tfm::Tensor other_ReLU_derivative(other);
	other_ReLU_derivative.ReLU_derivative();
	return this->multiply_elementwise(other_ReLU_derivative);
}


void tfm::Tensor::softmax() {
	if (tfm::Device::device_count > 0) {
		cuda_softmax(*this);
	}
	else {
		cpu_softmax(*this);
	}
}


void tfm::Tensor::softmax_backward(const tfm::Tensor& softmax_output) {
	if (tfm::Device::device_count > 0) {
		cuda_softmax_backward(*this, softmax_output);
	}
	else {
		cpu_softmax_backward(*this, softmax_output);
	}
}


void tfm::Tensor::sq() {
	if (tfm::Device::device_count > 0) {
		cuda_sq(*this);
	}
	else {
		cpu_sq(*this);
	}
}


void tfm::Tensor::sqrt() {
	if (tfm::Device::device_count > 0) {
		cuda_sqrt(*this);
	}
	else {
		cpu_sqrt(*this);
	}
}


tfm::Tensor tfm::Tensor::multiply(const tfm::Tensor& other, bool transposeThis, bool transposeOther) const {
	if (tfm::Device::device_count > 0) {
		return cuda_mat_mult(*this, other, transposeThis, transposeOther);
	}
	else {
		return cpu_mat_mult(*this, other, transposeThis, transposeOther);
	}
}


tfm::Tensor tfm::Tensor::multiply_elementwise(const Tensor& other) const {
	if (tfm::Device::device_count > 0) {
		return cuda_mat_mult_elementwise(*this, other);
	}
	else {
		return cpu_mat_mult_elementwise(*this, other);
	}
}


tfm::Tensor tfm::Tensor::divide_elementwise(const Tensor& other) const {
	if (tfm::Device::device_count > 0) {
		return cuda_mat_div_elementwise(*this, other);
	}
	else {
		return cpu_mat_div_elementwise(*this, other);
	}
}


tfm::Tensor tfm::Tensor::sum_along_axis(size_t axis) const {
	if (tfm::Device::device_count > 0) {
		return cuda_mat_add_along_axis(*this, axis);
	}
	else {
		return cpu_mat_add_along_axis(*this, axis);
	}
}


tfm::Tensor tfm::Tensor::operator+(const Tensor& other) const {
	if (tfm::Device::device_count > 0) {
		return cuda_mat_add(*this, other);
	}
	else {
		return cpu_mat_add(*this, other);
	}
}


tfm::Tensor& tfm::Tensor::operator+=(const Tensor& other) {
	if (tfm::Device::device_count > 0) {
		cuda_mat_add_inplace(*this, other);
	}
	else {
		cpu_mat_add_inplace(*this, other);
	}

	return *this;
}


tfm::Tensor tfm::Tensor::operator-(const Tensor& other) const {
	if (tfm::Device::device_count > 0) {
		return cuda_mat_sub(*this, other);
	}
	else {
		return cpu_mat_sub(*this, other);
	}
}


tfm::Tensor tfm::Tensor::operator*(const Tensor& other) const {
	if (tfm::Device::device_count > 0) {
		return cuda_mat_mult(*this, other);
	}
	else {
		return cpu_mat_mult(*this, other);
	}
}


tfm::Tensor tfm::Tensor::operator*(float val) const {
	if (tfm::Device::device_count > 0) {
		return cuda_mat_mult(*this, val);
	}
	else {
		return cpu_mat_mult(*this, val);
	}
}


tfm::Tensor tfm::Tensor::operator/(float val) const {
	if (tfm::Device::device_count > 0) {
		return cuda_mat_div(*this, val);
	}
	else {
		return cpu_mat_div(*this, val);
	}
}
