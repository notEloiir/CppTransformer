#include <stdio.h>
#include <backend/cpu/cpu_matrix_math.h>
#include <stdexcept>


float getElementTransposed(const tfm::Tensor& tensor, size_t col, size_t row, bool transpose) {
	if (transpose)
		return tensor[row][col];
	return tensor[col][row];
}

float fpow2(float x) {
	return x * x;
}


tfm::Tensor cpuMatAdd(const tfm::Tensor& A, const tfm::Tensor& B) {
	size_t cols = A.cols() < B.cols() ? A.cols() : B.cols();
	size_t rows = A.rows() < B.rows() ? A.rows() : B.rows();

	// if B is a vector, broadcast
	if (B.isVector()) {
		cols = A.cols();
	}

	tfm::Device device(tfm::DeviceType::CPU);
	const_cast<tfm::Tensor&>(A).moveTo(device);
	const_cast<tfm::Tensor&>(B).moveTo(device);
	tfm::Tensor C(cols, rows, device);

	if (B.isVector()) {
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


tfm::Tensor cpuMatMult(const tfm::Tensor& A, const tfm::Tensor& B, bool transposeA, bool transposeB) {
	size_t m = !transposeA ? A.rows() : A.cols();
	size_t n = !transposeB ? B.cols() : B.rows();
	size_t k = !transposeA ? A.cols() : A.rows();
	size_t k_check = !transposeB ? B.rows() : B.cols();

	if (k != k_check) {
		char message[128];
		snprintf(message, 128, "Matrices have incompatible dimensions for multiplication: (%zu, %zu), (%zu, %zu)", k, m, n, k_check);
		throw std::runtime_error(message);
	}

	tfm::Device device(tfm::DeviceType::CPU);
	const_cast<tfm::Tensor&>(A).moveTo(device);
	const_cast<tfm::Tensor&>(B).moveTo(device);
	tfm::Tensor C(n, m, device);

	for (size_t n_i = 0; n_i < n; n_i++) {
		for (size_t m_i = 0; m_i < m; m_i++) {
			C[n_i][m_i] = 0.0f;
			for (size_t k_i = 0; k_i < k; k_i++) {
				C[n_i][m_i] += getElementTransposed(A, k_i, m_i, transposeA) * getElementTransposed(B, n_i, k_i, transposeB);
			}
		}
	}

	return C;
}


tfm::Tensor cpuMatMultBLAS1(const tfm::Tensor& A, float val) {
	tfm::Tensor res(A.cols(), A.rows(), A.device());

	for (size_t col = 0; col < res.cols(); col++) {
		for (size_t row = 0; row < res.rows(); row++) {
			res[col][row] = A[col][row] * val;
		}
	}

	return res;
}


void cpuNormalizeMatrix(tfm::Tensor& matrix) {
	matrix.moveTo(tfm::Device(tfm::DeviceType::CPU));

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
			var += fpow2(matrix[col][row] - mean);
		}
		var /= matrix.cols();
		float stddev = sqrtf(var);

		for (size_t col = 0; col < matrix.cols(); col++) {
			matrix[col][row] = (matrix[col][row] - mean) / stddev * gamma + beta;
		}
	}

}

void cpuReLU(tfm::Tensor& matrix) {
	matrix.moveTo(tfm::Device(tfm::DeviceType::CPU));

	for (size_t col = 0; col < matrix.cols(); col++) {
		for (size_t row = 0; row < matrix.rows(); row++) {
			matrix[col][row] = matrix[col][row] > 0 ? matrix[col][row] : 0;
		}
	}
}


void cpuSoftmax(tfm::Tensor& matrix) {
	for (size_t col = 0; col < matrix.cols(); col++) {
		float maxVal = -FLT_MAX;
		for (size_t row = 0; row < matrix.rows(); row++) {
			if (maxVal < matrix[col][row]) {
				maxVal = matrix[col][row];
			}
		}

		float sum_exp = 0.0f;
		for (size_t row = 0; row < matrix.rows(); row++) {
			matrix[col][row] = std::exp(matrix[col][row] - maxVal);
			sum_exp += matrix[col][row];
		}

		for (size_t row = 0; row < matrix.rows(); row++) {
			matrix[col][row] /= sum_exp;
		}
	}

	return;
}

