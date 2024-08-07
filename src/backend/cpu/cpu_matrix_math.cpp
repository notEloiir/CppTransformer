#include <stdio.h>
#include <backend/cpu/cpu_matrix_math.h>


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

	tfm::Tensor C(cols, rows, A.device());

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

	tfm::Tensor C(n, m, A.device());

	for (size_t o = 0; o < m; o++) {
		for (size_t p = 0; p < n; p++) {
			C[p][o] = 0;
			for (size_t r = 0; r < k; r++) {
				C[p][o] += getElementTransposed(A, r, o, transposeA) * getElementTransposed(B, p, r, transposeB);
			}
		}
	}

	return C;
}


void cpuNormalizeMatrix(tfm::Tensor& matrix) {


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
			matrix[col][row] = matrix[col][row] - mean / stddev * gamma + beta;
		}
	}

}

void cpuReLU(tfm::Tensor& matrix) {
	for (size_t col = 0; col < matrix.cols(); col++) {
		for (size_t row = 0; row < matrix.rows(); row++) {
			matrix[col][row] = matrix[col][row] > 0 ? matrix[col][row] : 0;
		}
	}
}

