#include <doctest.h>

#include <backend/cpu/cpu_tensor_operations.h>


TEST_CASE("cpu_normalize_matrix") {
	tfm::Tensor t(2, 3, tfm::Device(tfm::DeviceType::CPU));
	t.init_weights();
	t.init_bias();
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

	cpu_normalize_matrix(t);

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

	cpu_normalize_matrix(t);

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		CHECK_EQ(t[i / t.rows()][i % t.rows()], doctest::Approx(expected[i / expected.rows()][i % expected.rows()]));
	}
}

TEST_CASE("cpu_ReLU") {
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

	cpu_ReLU(t);

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		CHECK_EQ(t[i / t.rows()][i % t.rows()], doctest::Approx(expected[i / expected.rows()][i % expected.rows()]));
	}
}

TEST_CASE("cpu_softmax") {
	tfm::Tensor t(2, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor expected(2, 3, tfm::Device(tfm::DeviceType::CPU));

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		t[i / t.rows()][i % t.rows()] = i * i;
	}

	expected[0][0] = 0.0171478f;
	expected[0][1] = 0.0466126f;
	expected[0][2] = 0.93624f;
	expected[1][0] = 0.0f;
	expected[1][1] = 0.000123f;
	expected[1][2] = 0.999876f;

	cpu_softmax(t);

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		CHECK_EQ(t[i / t.rows()][i % t.rows()], doctest::Approx(expected[i / expected.rows()][i % expected.rows()]));
	}
}
