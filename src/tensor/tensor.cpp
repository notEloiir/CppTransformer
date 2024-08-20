
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>

#include <tensor/tensor.h>
#include <backend/cuda/cuda_matrix_math.cuh>
#include <backend/cpu/cpu_matrix_math.h>


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
		v.data2D_ = (float**)realloc((void*)v.data2D_, v.cols_ * sizeof(float*));
	}

	for (size_t col = 0; col < v.cols(); col++) {
		v.data2D_[col] = data2D_[colIds[col]];
	}

	return v;
}


tfm::Tensor tfm::Tensor::nonOwningCopy(size_t colOffset, size_t cols) const {
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


tfm::Tensor tfm::Tensor::multiply(const tfm::Tensor& other, bool transposeThis, bool transposeOther) {
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


std::ostream& tfm::operator<<(std::ostream& os, const Tensor& t) {
	os << "Tensor size [" << t.cols() << ", " << t.rows() << "]\n";
	for (size_t row = 0; row < t.rows(); row++) {
		for (size_t col = 0; col < t.cols(); col++) {
			os << t[col][row] << ' ';
		}
		os << '\n';
	}

	return os;
}


