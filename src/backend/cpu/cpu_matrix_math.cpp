#include <stdio.h>
#include <stdexcept>

#include <backend/cpu/cpu_matrix_math.h>


float get_element_transposed(const tfm::Tensor& tensor, size_t col, size_t row, bool transpose) {
	if (transpose)
		return tensor[row][col];
	return tensor[col][row];
}

inline float pow2f(float x) {
	return x * x;
}


tfm::Tensor cpu_mat_add(const tfm::Tensor& A, const tfm::Tensor& B) {
	size_t cols = A.cols() < B.cols() ? A.cols() : B.cols();
	size_t rows = A.rows() < B.rows() ? A.rows() : B.rows();

	// if B is a vector, broadcast
	if (B.is_vector()) {
		cols = A.cols();
	}

	tfm::Tensor C(cols, rows, tfm::Device(tfm::DeviceType::CPU));

	if (B.is_vector()) {
		for (size_t col = 0; col < cols; col++) {
			for (size_t row = 0; row < rows; row++) {
				C[col][row] = A[col][row] + B[0][row];
			}
		}
	}
	else {
		for (size_t col = 0; col < cols; col++) {
			for (size_t row = 0; row < rows; row++) {
				C[col][row] = A[col][row] + B[col][row];
			}
		}
	}

	return C;
}


void cpu_mat_add_inplace(tfm::Tensor& A, const tfm::Tensor& B) {
	size_t cols = A.cols() < B.cols() ? A.cols() : B.cols();
	size_t rows = A.rows() < B.rows() ? A.rows() : B.rows();

	// if B is a vector, broadcast
	if (B.is_vector()) {
		cols = A.cols();
	}

	if (B.is_vector()) {
		for (size_t col = 0; col < cols; col++) {
			for (size_t row = 0; row < rows; row++) {
				A[col][row] += B[0][row];
			}
		}
	}
	else {
		for (size_t col = 0; col < cols; col++) {
			for (size_t row = 0; row < rows; row++) {
				A[col][row] += B[col][row];
			}
		}
	}

	return;
}


tfm::Tensor cpu_mat_add_along_axis(const tfm::Tensor& A, size_t axis) {
	if (axis > 1) {
		fprintf(stderr, "cpu_mat_add_along_axis axis > 1 not supported");
		exit(EXIT_FAILURE);
	}

	size_t cols = axis == 0 ? 1 : A.cols();
	size_t rows = axis == 0 ? A.rows() : 1;

	tfm::Tensor res(cols, rows, A.device());

	if (axis == 0) {
		for (size_t col = 0; col < res.cols(); col++) {
			for (size_t row = 0; row < res.rows(); row++) {
				res[0][row] += A[col][row];
			}
		}
	}
	else {
		for (size_t col = 0; col < res.cols(); col++) {
			for (size_t row = 0; row < res.rows(); row++) {
				res[col][0] += A[col][row];
			}
		}
	}

	return res;
}


tfm::Tensor cpu_mat_sub(const tfm::Tensor& A, const tfm::Tensor& B) {
	size_t cols = A.cols() < B.cols() ? A.cols() : B.cols();
	size_t rows = A.rows() < B.rows() ? A.rows() : B.rows();

	// if B is a vector, broadcast
	if (B.is_vector()) {
		cols = A.cols();
	}

	tfm::Tensor C(cols, rows, tfm::Device(tfm::DeviceType::CPU));

	if (B.is_vector()) {
		for (size_t col = 0; col < cols; col++) {
			for (size_t row = 0; row < rows; row++) {
				C[col][row] = A[col][row] - B[0][row];
			}
		}
	}
	else {
		for (size_t col = 0; col < cols; col++) {
			for (size_t row = 0; row < rows; row++) {
				C[col][row] = A[col][row] - B[col][row];
			}
		}
	}

	return C;
}


tfm::Tensor cpu_mat_mult(const tfm::Tensor& A, const tfm::Tensor& B, bool transpose_A, bool transpose_B) {
	size_t m = !transpose_A ? A.rows() : A.cols();
	size_t n = !transpose_B ? B.cols() : B.rows();
	size_t k = !transpose_A ? A.cols() : A.rows();
	size_t k_check = !transpose_B ? B.rows() : B.cols();

	if (k != k_check) {
		char message[128];
		snprintf(message, 128, "Matrices have incompatible dimensions for multiplication: (%zu, %zu), (%zu, %zu)", k, m, n, k_check);
		throw std::runtime_error(message);
	}

	tfm::Tensor C(n, m, tfm::Device(tfm::DeviceType::CPU));

	for (size_t n_i = 0; n_i < n; n_i++) {
		for (size_t m_i = 0; m_i < m; m_i++) {
			C[n_i][m_i] = 0.0f;
			for (size_t k_i = 0; k_i < k; k_i++) {
				C[n_i][m_i] += get_element_transposed(A, k_i, m_i, transpose_A) * get_element_transposed(B, n_i, k_i, transpose_B);
			}
		}
	}

	return C;
}


tfm::Tensor cpu_mat_mult_elementwise(const tfm::Tensor& A, const tfm::Tensor& B) {
	size_t cols = A.cols() < B.cols() ? A.cols() : B.cols();
	size_t rows = A.rows() < B.rows() ? A.rows() : B.rows();

	tfm::Tensor C(cols, rows, tfm::Device(tfm::DeviceType::CPU));

	for (size_t col = 0; col < cols; col++) {
		for (size_t row = 0; row < rows; row++) {
			C[col][row] = A[col][row] * B[col][row];
		}
	}

	return C;
}


tfm::Tensor cpu_mat_mult(const tfm::Tensor& A, float val) {
	tfm::Tensor res(A.cols(), A.rows(), A.device());

	for (size_t col = 0; col < res.cols(); col++) {
		for (size_t row = 0; row < res.rows(); row++) {
			res[col][row] = A[col][row] * val;
		}
	}

	return res;
}


tfm::Tensor cpu_mat_div_elementwise(const tfm::Tensor& A, const tfm::Tensor& B) {
	tfm::Tensor res(A.cols(), A.rows(), A.device());

	for (size_t col = 0; col < res.cols(); col++) {
		for (size_t row = 0; row < res.rows(); row++) {
			res[col][row] = A[col][row] / B[col][row];
		}
	}

	return res;
}


tfm::Tensor cpu_mat_div(const tfm::Tensor& A, float val) {
	tfm::Tensor res(A.cols(), A.rows(), A.device());

	for (size_t col = 0; col < res.cols(); col++) {
		for (size_t row = 0; row < res.rows(); row++) {
			res[col][row] = A[col][row] / val;
		}
	}

	return res;
}


void cpu_sq(tfm::Tensor& matrix) {
	for (size_t col = 0; col < matrix.cols(); col++) {
		for (size_t row = 0; row < matrix.rows(); row++) {
			matrix[col][row] = pow2f(matrix[col][row]);
		}
	}
}


void cpu_sqrt(tfm::Tensor& matrix) {
	for (size_t col = 0; col < matrix.cols(); col++) {
		for (size_t row = 0; row < matrix.rows(); row++) {
			matrix[col][row] = sqrtf(matrix[col][row]);
		}
	}
}

