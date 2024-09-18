#include <doctest.h>

#include <backend/cpu/cpu_matrix_math.h>


TEST_CASE("cpu_mat_add") {
	tfm::Tensor t0(2, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor t1(2, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor expected(2, 3, tfm::Device(tfm::DeviceType::CPU));

	for (size_t i = 0; i < t0.cols() * t0.rows(); i++) {
		t0[i / t0.rows()][i % t0.rows()] = i * i;
		t1[i / t1.rows()][i % t1.rows()] = 2.0f * i - 5.0f;
		expected[i / expected.rows()][i % expected.rows()] = i * i + 2.0f * i - 5.0f;
	}

	tfm::Tensor res = cpu_mat_add(t0, t1);

	for (size_t i = 0; i < t0.cols() * t0.rows(); i++) {
		CHECK_EQ(res[i / res.rows()][i % res.rows()], expected[i / expected.rows()][i % expected.rows()]);
	}
}

TEST_CASE("cpu_mat_add") {
	tfm::Tensor t0(2, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor expected0(1, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor expected1(2, 1, tfm::Device(tfm::DeviceType::CPU));

	for (size_t i = 0; i < t0.cols() * t0.rows(); i++) {
		t0[i / t0.rows()][i % t0.rows()] = i * i;
	}

	expected0[0][0] = 0.0f + 9.0f;
	expected0[0][1] = 1.0f + 16.0f;
	expected0[0][2] = 4.0f + 25.0f;

	tfm::Tensor res0 = cpu_mat_add_along_axis(t0, 0);

	for (size_t i = 0; i < res0.cols() * res0.rows(); i++) {
		CHECK_EQ(res0[i / res0.rows()][i % res0.rows()], doctest::Approx(expected0[i / expected0.rows()][i % expected0.rows()]));
	}

	expected1[0][0] = 0.0f + 1.0f + 4.0f;
	expected1[1][0] = 9.0f + 16.0f + 25.0f;

	tfm::Tensor res1 = cpu_mat_add_along_axis(t0, 1);

	for (size_t i = 0; i < res1.cols() * res1.rows(); i++) {
		CHECK_EQ(res1[i / res1.rows()][i % res1.rows()], doctest::Approx(expected1[i / expected1.rows()][i % expected1.rows()]));
	}
}

TEST_CASE("cpu_mat_add_inplace") {
	tfm::Tensor t0(2, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor t1(2, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor expected(2, 3, tfm::Device(tfm::DeviceType::CPU));

	for (size_t i = 0; i < t0.cols() * t0.rows(); i++) {
		t0[i / t0.rows()][i % t0.rows()] = i * i;
		t1[i / t1.rows()][i % t1.rows()] = 2.0f * i - 5.0f;
		expected[i / expected.rows()][i % expected.rows()] = i * i + 2.0f * i - 5.0f;
	}

	cpu_mat_add_inplace(t0, t1);

	for (size_t i = 0; i < t0.cols() * t0.rows(); i++) {
		CHECK_EQ(t0[i / t0.rows()][i % t0.rows()], expected[i / expected.rows()][i % expected.rows()]);
	}
}

TEST_CASE("cpu_mat_mult") {
	tfm::Tensor t0(1, 2, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor t1(4, 2, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor expected(4, 1, tfm::Device(tfm::DeviceType::CPU));

	t0[0][0] = -2.0f;
	t0[0][1] = 5.0f;
	for (size_t i = 0; i < t1.cols() * t1.rows(); i++) {
		t1[i % t1.cols()][i / t1.cols()] = i;
	}

	// check if exception thrown on size mismatch
	CHECK_THROWS(cpu_mat_mult(t0, t1, false, false));
	CHECK_THROWS(cpu_mat_mult(t0, t1, false, true));
	CHECK_NOTHROW(cpu_mat_mult(t0, t1, true, false));
	CHECK_THROWS(cpu_mat_mult(t0, t1, true, true));

	expected[0][0] = 20.0f;
	expected[1][0] = 23.0f;
	expected[2][0] = 26.0f;
	expected[3][0] = 29.0f;

	tfm::Tensor res = cpu_mat_mult(t0, t1, true, false);

	CHECK_EQ(res.cols(), expected.cols());
	CHECK_EQ(res.rows(), expected.rows());

	for (size_t i = 0; i < res.cols() * res.rows(); i++) {
		CHECK_EQ(res[i / res.rows()][i % res.rows()], expected[i / expected.rows()][i % expected.rows()]);
	}
}

TEST_CASE("cpu_mat_mult BLAS1") {
	tfm::Tensor t(2, 3, tfm::Device(tfm::DeviceType::CPU));
	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		t[i % t.cols()][i / t.cols()] = i;
	}

	tfm::Tensor res = cpu_mat_mult(t, 2.5f);

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		CHECK_EQ(res[i / res.rows()][i % res.rows()], doctest::Approx(2.5f * t[i / t.rows()][i % t.rows()]));
	}
}
