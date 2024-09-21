
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <cuda_runtime.h>

#include <tensor/tensor.h>
#include <cuda_utils.h>


tfm::Tensor tfm::Tensor::concatenate(const std::vector<tfm::Tensor>& tensors, size_t dim) {
	if (tensors.empty()) {
		return tfm::Tensor();
	}
	if (dim >= 2) {
		fprintf(stderr, "Concatenating tensors along dim > 1 not supported.");
		exit(EXIT_FAILURE);
	}
	for (size_t i = 0; i < tensors.size() - 1; i++) {
		if ((dim == 0 && tensors[i].rows() != tensors[i + 1].rows()) || (dim == 1 && tensors[i].cols() != tensors[i + 1].cols())) {
			fprintf(stderr, "Concatenating tensors along dim > 1 not supported.");
			exit(EXIT_FAILURE);
		}
		if (tensors[i].device() != tensors[i + 1].device()) {
			fprintf(stderr, "Concatenated tensors must be on the same device.");
			exit(EXIT_FAILURE);
		}
	}

	size_t cols, rows;
	if (dim == 0) {
		cols = 0;
		rows = tensors[0].rows();
		for (size_t i = 0; i < tensors.size(); i++) {
			cols += tensors[i].cols();
		}
	}
	else {  // dim == 1
		cols = tensors[0].cols();
		rows = 0;
		for (size_t i = 0; i < tensors.size(); i++) {
			rows += tensors[i].rows();
		}
	}

	tfm::Tensor t(cols, rows, tensors[0].device());

	if (t.device().is_CUDA()) {
		// copy GPU side
		if (dim == 0) {
			size_t start_pos = 0;
			for (size_t i = 0; i < tensors.size(); i++) {
				check_cuda_error(cudaMemcpy(t.data() + start_pos , tensors[i].data(), tensors[i].cols() * tensors[i].rows() * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy CUDA data");
				start_pos += tensors[i].cols() * tensors[i].rows();
			}
		}
		else {  // dim == 1
			size_t row_start = 0;
			for (size_t i = 0; i < tensors.size(); i++) {
				for (size_t col = 0; col < tensors[i].cols(); col++) {
					check_cuda_error(cudaMemcpy(t.data() + t.rows() * col + row_start, tensors[i].data() + tensors[i].rows() * col, tensors[i].rows() * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy CUDA data");
				}
				row_start += tensors[i].rows();
			}
		}
	}
	else {  // device_.is_CPU()
		// copy RAM side
		if (dim == 0) {
			size_t start_pos = 0;
			for (size_t i = 0; i < tensors.size(); i++) {
				if (tensors[i].is_data_continuous_) {
					memcpy(t.data() + start_pos, tensors[i].data(), tensors[i].cols() * tensors[i].rows() * sizeof(float));
				}
				else {
					for (size_t col = 0; col < tensors[i].cols(); col++) {
						memcpy(t.data() + start_pos + col * t.rows(), tensors[i].col_data(col), tensors[i].rows() * sizeof(float));
					}
				}
				start_pos += tensors[i].cols() * tensors[i].rows();
			}
		}
		else {  // dim == 1
			size_t row_start = 0;
			for (size_t i = 0; i < tensors.size(); i++) {
				for (size_t col = 0; col < tensors[i].cols(); col++) {
					memcpy(t.data() + row_start + col * t.rows(), tensors[i].col_data(col), tensors[i].rows() * sizeof(float));
				}
				row_start += tensors[i].rows();
			}
		}
	}

	return t;
}



tfm::Tensor tfm::Tensor::subtensor(const tfm::Tensor& other, size_t cols, size_t rows, size_t col_offset, size_t row_offset) {
	if (col_offset + cols > other.cols() || row_offset + rows > other.rows()) {
		fprintf(stderr, "col_offset + cols > other.cols() || row_offset + rows > other.rows()");
		exit(EXIT_FAILURE);
	}

	tfm::Tensor t(cols, rows, other.device());

	if (t.device().is_CUDA()) {
		// copy GPU side
		for (size_t col = 0; col < cols; col++) {
			check_cuda_error(cudaMemcpy(t.data() + rows * col, other.data() + other.rows() * (col + col_offset) + row_offset, rows * sizeof(float), cudaMemcpyDeviceToDevice), "Failed to copy CUDA data");
		}
	}
	else {  // device_.is_CPU()
		// copy RAM side
		for (size_t col = 0; col < cols; col++) {
			memcpy(t.data() + rows * col, other.col_data(col + col_offset) + row_offset, rows * sizeof(float));
		}
	}

	return t;
}


float* tfm::Tensor::data() const {
	if (device_.is_CPU()) {
		return data_;
	}
	else {
		return data_cuda_;
	}
}


float* tfm::Tensor::col_data(size_t col) const {
	if (device_.is_CPU()) {
		return data_2D_[col];
	}
	else {
		return data_cuda_ + col * rows_;
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

