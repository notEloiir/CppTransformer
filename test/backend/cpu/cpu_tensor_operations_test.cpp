#include <doctest.h>

#include <backend/cpu/cpu_tensor_operations.h>


TEST_CASE("cpu_normalize_matrix") {
	tfm::Tensor t(2, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor weights(1, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor bias(1, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor expected(2, 3, tfm::Device(tfm::DeviceType::CPU));

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		t[i / t.rows()][i % t.rows()] = i * i;
	}
	weights.fill(1.0f);
	bias.fill(0.0f);

	expected[0][0] = -1.0f;
	expected[0][1] = -1.0f;
	expected[0][2] = -1.0f;
	expected[1][0] = 1.0f;
	expected[1][1] = 1.0f;
	expected[1][2] = 1.0f;

	cpu_normalize_matrix(t, weights, bias);

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		CHECK_EQ(t[i / t.rows()][i % t.rows()], doctest::Approx(expected[i / expected.rows()][i % expected.rows()]));
	}

	// original matrix, different weights and biases

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		t[i / t.rows()][i % t.rows()] = i * i;
	}

	weights[0][0] = 3.0f;
	weights[0][1] = -5.0f;
	weights[0][2] = 1.0f;
	bias[0][0] = 3.0f;
	bias[0][1] = -0.7f;
	bias[0][2] = 2.0f;

	expected[0][0] = 0.0f;
	expected[0][1] = 4.3f;
	expected[0][2] = 1.0f;
	expected[1][0] = 6.0f;
	expected[1][1] = -5.7f;
	expected[1][2] = 3.0f;

	cpu_normalize_matrix(t, weights, bias);

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
