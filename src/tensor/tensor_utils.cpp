
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>

#include <tensor/tensor.h>


static tfm::Tensor concatenate(const std::vector<tfm::Tensor>& tensors, size_t dim) {
	if (tensors.empty()) {
		return;
	}
	if (dim >= 2) {
		fprintf(stderr, "Concatenating tensors along dim > 1 not supported.");
		return;
	}
	for (size_t i = 0; i < tensors.size() - 1; i++) {
		if (dim == 0 && tensors[i].rows() != tensors[i + 1].rows()) {
			fprintf(stderr, "Concatenating tensors along dim > 1 not supported.");
			return;
		}
	}

	// TODO: implement
	fprintf(stderr, "not implemented");
}



static tfm::Tensor subtensor(const tfm::Tensor& other, size_t cols, size_t rows, size_t colOffset, size_t rowOffset) {
	if (colOffset + cols > other.cols() || rowOffset + rows > other.rows()) {
		fprintf(stderr, "colOffset + cols > other.cols() || rowOffset + rows > other.rows()");
		exit(EXIT_FAILURE);
	}

	// TODO: implement
	fprintf(stderr, "not implemented");
}


float* tfm::Tensor::data() const {
	if (device_.isCPU()) {
		return data_;
	}
	else {
		return dataCuda_;
	}
}

float* tfm::Tensor::colData(size_t col) const {
	if (device_.isCPU()) {
		return data2D_[col];
	}
	else {
		return dataCuda_ + col * rows_;
	}
}

float* tfm::Tensor::weights() const {
	if (device_.isCPU()) {
		return weights_;
	}
	else {
		return weightsCuda_;
	}
}

float* tfm::Tensor::bias() const {
	if (device_.isCPU()) {
		return bias_;
	}
	else {
		return biasCuda_;
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

