
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>

#include <tensor/tensor.h>
#include <backend/cuda/cuda_matrix_math.cuh>
#include <backend/cpu/cpu_matrix_math.h>


void tfm::Tensor::zeroes() {
	if (isDataContinuous_) {
		std::fill(data_, data_ + cols_ * rows_, 0.0f);
	}
	else {
		for (size_t col = 0; col < cols(); col++) {
			std::fill(data2D_[col], data2D_[col] + rows_, 0.0f);
		}
	}
}


void tfm::Tensor::ones() {
	if (isDataContinuous_) {
		std::fill(data_, data_ + cols_ * rows_, 1.0f);
	}
	else {
		for (size_t col = 0; col < cols(); col++) {
			std::fill(data2D_[col], data2D_[col] + rows_, 1.0f);
		}
	}
}


void tfm::Tensor::diag() {
	zeroes();
	for (size_t i = 0; i < cols() && i < rows(); i++) {
		data2D_[i][i] = 1.0f;
	}
}


void tfm::Tensor::random() {
	for (size_t col = 0; col < cols(); col++) {
		for (size_t row = 0; row < rows(); row++) {
			data2D_[col][row] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2));
		}
	}
}


void tfm::Tensor::normalize() {
	if (tfm::Device::deviceCount > 0) {
		cudaNormalizeMatrix(*this);
	}
	else {
		cpuNormalizeMatrix(*this);
	}
}


void tfm::Tensor::ReLU() {
	if (tfm::Device::deviceCount > 0) {
		cudaReLU(*this);
	}
	else {
		cpuReLU(*this);
	}
}


void tfm::Tensor::softmax() {
	if (tfm::Device::deviceCount > 0) {
		cudaSoftmax(*this);
	}
	else {
		cpuSoftmax(*this);
	}
}


tfm::Tensor tfm::Tensor::multiply(const tfm::Tensor& other, bool transposeThis, bool transposeOther) const {
	if (tfm::Device::deviceCount > 0) {
		return cudaMatMult(*this, other, transposeThis, transposeOther);
	}
	else {
		return cpuMatMult(*this, other, transposeThis, transposeOther);
	}
}


tfm::Tensor tfm::Tensor::operator+(const Tensor& other) const {
	if (tfm::Device::deviceCount > 0) {
		return cudaMatAdd(*this, other);
	}
	else {
		return cpuMatAdd(*this, other);
	}
}


tfm::Tensor tfm::Tensor::operator*(const Tensor& other) const {
	if (tfm::Device::deviceCount > 0) {
		return cudaMatMult(*this, other);
	}
	else {
		return cpuMatMult(*this, other);
	}
}


tfm::Tensor tfm::Tensor::operator*(float val) const {
	if (tfm::Device::deviceCount > 0) {
		return cudaMatMultBLAS1(*this, val);
	}
	else {
		return cpuMatMultBLAS1(*this, val);
	}
}


