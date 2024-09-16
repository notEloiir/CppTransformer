#include <stdio.h>
#include <backend/cpu/cpu_matrix_math.h>
#include <stdexcept>


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


void cpu_normalize_matrix(tfm::Tensor& matrix) {
	for (size_t row = 0; row < matrix.rows(); row++) {
		float gamma = matrix.weights()[row];
		float beta = matrix.bias()[row];

		float mean = 0.0f;
		float var = 0.0f;

		for (size_t col = 0; col < matrix.cols(); col++) {
			mean += matrix[col][row];
		}
		mean /= matrix.cols();

		for (size_t col = 0; col < matrix.cols(); col++) {
			var += pow2f(matrix[col][row] - mean);
		}
		var /= matrix.cols();
		float stddev = sqrtf(var);

		for (size_t col = 0; col < matrix.cols(); col++) {
			matrix[col][row] = (matrix[col][row] - mean) / stddev * gamma + beta;
		}
	}

}


void cpu_normalize_matrix_backward(tfm::Tensor& normalize_output, const tfm::Tensor& grad_output) {
	for (size_t row = 0; row < normalize_output.rows(); row++) {
		// Get gamma and beta
		float gamma = normalize_output.weights()[row];
		float beta = normalize_output.bias()[row];

		// Recalculate mean and variance
		float mean = 0.0f;
		float var = 0.0f;

		for (size_t col = 0; col < normalize_output.cols(); col++) {
			mean += normalize_output[col][row];
		}
		mean /= normalize_output.cols();

		for (size_t col = 0; col < normalize_output.cols(); col++) {
			var += pow2f(normalize_output[col][row] - mean);
		}
		var /= normalize_output.cols();
		float stddev = sqrtf(var);

		// Calculate gradients for gamma and beta
		float grad_gamma = 0.0f;
		float grad_beta = 0.0f;

		for (size_t col = 0; col < normalize_output.cols(); col++) {
			grad_gamma += grad_output[col][row] * (normalize_output[col][row] - beta) / gamma;
			grad_beta += grad_output[col][row];
		}

		// Calculate gradients wrt normalized input
		float grad_mean = 0.0f;
		float grad_var = 0.0f;

		for (size_t col = 0; col < normalize_output.cols(); col++) {
			// Gradient wrt the normalized output
			float grad_norm = grad_output[col][row] * gamma;

			// Accumulate gradients wrt variance and mean
			grad_var += grad_norm * (normalize_output[col][row] - mean) * -0.5f * powf(var + FLT_MIN, -1.5f);
			grad_mean += grad_norm * -1.0f / stddev;
		}

		// Gradient wrt input
		for (size_t col = 0; col < normalize_output.cols(); col++) {
			float grad_input = (grad_output[col][row] * gamma) / stddev;
			grad_input += grad_var * 2.0f * (normalize_output[col][row] - mean) / normalize_output.cols();
			grad_input += grad_mean / normalize_output.cols();

			normalize_output[col][row] = grad_input;
		}

		normalize_output.weights()[row] -= grad_gamma;
		normalize_output.bias()[row] -= grad_beta;
	}
}


void cpu_ReLU(tfm::Tensor& matrix) {
	for (size_t col = 0; col < matrix.cols(); col++) {
		for (size_t row = 0; row < matrix.rows(); row++) {
			matrix[col][row] = matrix[col][row] > 0 ? matrix[col][row] : 0;
		}
	}
}


void cpu_ReLU_derivative(tfm::Tensor& matrix) {
	for (size_t col = 0; col < matrix.cols(); col++) {
		for (size_t row = 0; row < matrix.rows(); row++) {
			matrix[col][row] = static_cast<float>(matrix[col][row] > 0);
		}
	}
}


void cpu_softmax(tfm::Tensor& matrix) {
	for (size_t col = 0; col < matrix.cols(); col++) {
		float max_val = -FLT_MAX;
		for (size_t row = 0; row < matrix.rows(); row++) {
			if (max_val < matrix[col][row]) {
				max_val = matrix[col][row];
			}
		}

		float sum_exp = 0.0f;
		for (size_t row = 0; row < matrix.rows(); row++) {
			matrix[col][row] = std::exp(matrix[col][row] - max_val);
			sum_exp += matrix[col][row];
		}

		for (size_t row = 0; row < matrix.rows(); row++) {
			matrix[col][row] /= sum_exp;
		}
	}

	return;
}


void cpu_softmax_backward(tfm::Tensor& softmax_output, const tfm::Tensor& grad_output) {
	tfm::Tensor grad_input(softmax_output.rows(), softmax_output.cols(), softmax_output.device());
	grad_input.fill(0.0f);

	for (size_t col = 0; col < softmax_output.cols(); col++) {
		float* softmax_col = softmax_output.col_data(col);
		float* grad_col = grad_output.col_data(col);

		// Calculate the dot product of the softmax vector with the gradient vector
		float dot_product = 0.0f;
		for (size_t row = 0; row < softmax_output.rows(); row++) {
			dot_product += softmax_col[row] * grad_col[row];
		}

		// Compute the gradient for each element in the column
		for (size_t row = 0; row < softmax_output.rows(); row++) {
			// For each element in the column, apply the softmax derivative formula:
			// grad_input = softmax * (grad_output - (softmax * sum(grad_output)))
			grad_input[col][row] = softmax_col[row] * (grad_col[row] - dot_product);
		}
	}

	softmax_output = std::move(grad_input);
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

