#include <doctest.h>
#include <backend/cpu/cpu_matrix_math.h>

TEST_CASE("CPU tensor addition") {
	tfm::Tensor t0(2, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor t1(2, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor expected(2, 3, tfm::Device(tfm::DeviceType::CPU));

	for (size_t i = 0; i < t0.cols() * t0.rows(); i++) {
		t0[i / t0.rows()][i % t0.rows()] = i * i;
		t1[i / t1.rows()][i % t1.rows()] = 2.0f * i - 5.0f;
		expected[i / expected.rows()][i % expected.rows()] = i * i + 2.0f * i - 5.0f;
	}

	tfm::Tensor res = cpuMatAdd(t0, t1);

	for (size_t i = 0; i < t0.cols() * t0.rows(); i++) {
		CHECK_EQ(res[i / res.rows()][i % res.rows()], expected[i / expected.rows()][i % expected.rows()]);
	}
}
TEST_CASE("CPU tensor multiplication") {
	tfm::Tensor t0(1, 2, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor t1(4, 2, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor expected(4, 1, tfm::Device(tfm::DeviceType::CPU));

	t0[0][0] = -2.0f;
	t0[0][1] = 5.0f;
	for (size_t i = 0; i < t1.cols() * t1.rows(); i++) {
		t1[i % t1.cols()][i / t1.cols()] = i;
	}

	CHECK_THROWS(cpuMatMult(t0, t1, false, false));
	CHECK_THROWS(cpuMatMult(t0, t1, false, true));
	CHECK_NOTHROW(cpuMatMult(t0, t1, true, false));
	CHECK_THROWS(cpuMatMult(t0, t1, true, true));

	expected[0][0] = 20.0f;
	expected[1][0] = 23.0f;
	expected[2][0] = 26.0f;
	expected[3][0] = 29.0f;

	tfm::Tensor res = cpuMatMult(t0, t1, true, false);

	CHECK_EQ(res.cols(), expected.cols());
	CHECK_EQ(res.rows(), expected.rows());

	for (size_t i = 0; i < res.cols() * res.rows(); i++) {
		CHECK_EQ(res[i / res.rows()][i % res.rows()], expected[i / expected.rows()][i % expected.rows()]);
	}
}
TEST_CASE("CPU tensor normalization") {
	tfm::Tensor t(2, 3, tfm::Device(tfm::DeviceType::CPU));
	t.initWeights();
	t.initBias();
	tfm::Tensor expected(2, 3, tfm::Device(tfm::DeviceType::CPU));

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		t[i / t.rows()][i % t.rows()] = i * i;
	}

	expected[0][0] = -1.0f;
	expected[0][1] = -1.0f;
	expected[0][2] = -1.0f;
	expected[1][0] = 1.0f;
	expected[1][1] = 1.0f;
	expected[1][2] = 1.0f;

	cpuNormalizeMatrix(t);

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		CHECK_EQ(t[i / t.rows()][i % t.rows()], doctest::Approx(expected[i / expected.rows()][i % expected.rows()]));
	}

	// original matrix, different weights and biases

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		t[i / t.rows()][i % t.rows()] = i * i;
	}

	t.weights()[0] = 3.0f;
	t.weights()[1] = -5.0f;
	t.weights()[2] = 1.0f;
	t.bias()[0] = 3.0f;
	t.bias()[1] = -0.7f;
	t.bias()[2] = 2.0f;

	expected[0][0] = 0.0f;
	expected[0][1] = 4.3f;
	expected[0][2] = 1.0f;
	expected[1][0] = 6.0f;
	expected[1][1] = -5.7f;
	expected[1][2] = 3.0f;

	cpuNormalizeMatrix(t);

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		CHECK_EQ(t[i / t.rows()][i % t.rows()], doctest::Approx(expected[i / expected.rows()][i % expected.rows()]));
	}
}
TEST_CASE("CPU tensor ReLU") {
	tfm::Tensor t(2, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor expected(2, 3, tfm::Device(tfm::DeviceType::CPU));

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		t[i / t.rows()][i % t.rows()] = sin(i);
	}

	expected[0][0] = 0.0f;
	expected[0][1] = 0.841471f;
	expected[0][2] = 0.909297f;
	expected[1][0] = 0.14112f;
	expected[1][1] = 0.0f;
	expected[1][2] = 0.0f;
	
	cpuReLU(t);

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		CHECK_EQ(t[i / t.rows()][i % t.rows()], doctest::Approx(expected[i / expected.rows()][i % expected.rows()]));
	}
}
