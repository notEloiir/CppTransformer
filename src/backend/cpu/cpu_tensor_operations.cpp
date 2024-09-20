#include <stdio.h>
#include <stdexcept>

#include <backend/cpu/cpu_tensor_operations.h>


inline float pow2f(float x) {
	return x * x;
}


void cpu_normalize_matrix(tfm::Tensor& matrix) {
	for (size_t row = 0; row < matrix.rows(); row++) {
		// Get gamma and beta
		float gamma = matrix.weights()[row];
		float beta = matrix.bias()[row];

		// Calculate mean and variance
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

		// Normalize values
		for (size_t col = 0; col < matrix.cols(); col++) {
			matrix[col][row] = (matrix[col][row] - mean) / stddev * gamma + beta;
		}
	}

}


void cpu_normalize_matrix_backward(tfm::Tensor& grad, const tfm::Tensor& normalize_input) {
	for (size_t row = 0; row < normalize_input.rows(); row++) {
		// Get gamma and beta
		float gamma = normalize_input.weights()[row];
		float beta = normalize_input.bias()[row];

		// Recalculate mean and variance
		float mean = 0.0f;
		float var = 0.0f;

		for (size_t col = 0; col < normalize_input.cols(); col++) {
			mean += normalize_input[col][row];
		}
		mean /= normalize_input.cols();

		for (size_t col = 0; col < normalize_input.cols(); col++) {
			var += pow2f(normalize_input[col][row] - mean);
		}
		var /= normalize_input.cols();
		float stddev = sqrtf(var);

		// Calculate gradients for gamma and beta
		float grad_gamma = 0.0f;
		float grad_beta = 0.0f;

		for (size_t col = 0; col < normalize_input.cols(); col++) {
			grad_gamma += grad[col][row] * (normalize_input[col][row] - beta) / gamma;
			grad_beta += grad[col][row];
		}

		// Calculate gradients wrt normalized input
		float grad_mean = 0.0f;
		float grad_var = 0.0f;

		for (size_t col = 0; col < normalize_input.cols(); col++) {
			// Gradient wrt the normalized output
			float grad_norm = grad[col][row] * gamma;

			// Accumulate gradients wrt variance and mean
			grad_var += grad_norm * (normalize_input[col][row] - mean) * -0.5f * powf(var + FLT_MIN, -1.5f);
			grad_mean += grad_norm * -1.0f / stddev;
		}

		// Gradient wrt input
		for (size_t col = 0; col < normalize_input.cols(); col++) {
			float grad_input_cell = (grad[col][row] * gamma) / stddev;
			grad_input_cell += grad_var * 2.0f * (normalize_input[col][row] - mean) / normalize_input.cols();
			grad_input_cell += grad_mean / normalize_input.cols();

			grad[col][row] = grad_input_cell;
		}

		grad.weights()[row] -= grad_gamma;
		grad.bias()[row] -= grad_beta;
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
		// Find max value in a col
		float max_val = -FLT_MAX;
		for (size_t row = 0; row < matrix.rows(); row++) {
			if (max_val < matrix[col][row]) {
				max_val = matrix[col][row];
			}
		}

		// For each value apply exp(val - max), -max to keep numerical stability
		float sum_exp = 0.0f;
		for (size_t row = 0; row < matrix.rows(); row++) {
			matrix[col][row] = std::exp(matrix[col][row] - max_val);
			sum_exp += matrix[col][row];
		}

		// Turn into probability
		for (size_t row = 0; row < matrix.rows(); row++) {
			matrix[col][row] /= sum_exp;
		}
	}

	return;
}


void cpu_softmax_backward(tfm::Tensor& grad, const tfm::Tensor& softmax_output) {
	for (size_t col = 0; col < softmax_output.cols(); col++) {
		float* softmax_col = softmax_output.col_data(col);
		float* grad_col = grad.col_data(col);

		// Calculate the dot product of the softmax vector with the gradient vector
		float dot_product = 0.0f;
		for (size_t row = 0; row < softmax_output.rows(); row++) {
			dot_product += softmax_col[row] * grad_col[row];
		}

		// Compute the gradient for each element in the column
		for (size_t row = 0; row < softmax_output.rows(); row++) {
			// For each element in the column, apply the softmax derivative formula:
			// grad = softmax * (grad - (softmax * sum(grad)))
			grad[col][row] = softmax_col[row] * (grad_col[row] - dot_product);
		}
	}
}
