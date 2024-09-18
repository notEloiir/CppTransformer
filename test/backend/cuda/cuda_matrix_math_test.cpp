#include <doctest.h>

#include <compiler_flags.h>
#include <backend/cuda/cuda_matrix_math.cuh>
#include <backend/cuda/cuda_tensor_operations.cuh>


#ifndef NO_CUDA

TEST_CASE("cuda_mat_add") {
	tfm::Tensor t0(2, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor t1(2, 3, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor expected(2, 3, tfm::Device(tfm::DeviceType::CPU));

	for (size_t i = 0; i < t0.cols() * t0.rows(); i++) {
		t0[i / t0.rows()][i % t0.rows()] = i * i;
		t1[i / t1.rows()][i % t1.rows()] = 2.0f * i - 5.0f;
		expected[i / expected.rows()][i % expected.rows()] = i * i + 2.0f * i - 5.0f;
	}

	tfm::Tensor res = cuda_mat_add(t0, t1);
	res.move_to(tfm::Device(tfm::DeviceType::CPU));

	for (size_t i = 0; i < t0.cols() * t0.rows(); i++) {
		CHECK_EQ(res[i / res.rows()][i % res.rows()], expected[i / expected.rows()][i % expected.rows()]);
	}
}

TEST_CASE("cuda_mat_mult") {
	tfm::Tensor t0(1, 2, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor t1(4, 2, tfm::Device(tfm::DeviceType::CPU));
	tfm::Tensor expected(4, 1, tfm::Device(tfm::DeviceType::CPU));

	t0[0][0] = -2.0f;
	t0[0][1] = 5.0f;
	for (size_t i = 0; i < t1.cols() * t1.rows(); i++) {
		t1[i % t1.cols()][i / t1.cols()] = i;
	}

	CHECK_THROWS(cuda_mat_mult(t0, t1, false, false));
	CHECK_THROWS(cuda_mat_mult(t0, t1, false, true));
	CHECK_NOTHROW(cuda_mat_mult(t0, t1, true, false));
	CHECK_THROWS(cuda_mat_mult(t0, t1, true, true));

	expected[0][0] = 20.0f;
	expected[1][0] = 23.0f;
	expected[2][0] = 26.0f;
	expected[3][0] = 29.0f;

	tfm::Tensor res = cuda_mat_mult(t0, t1, true, false);
	res.move_to(tfm::Device(tfm::DeviceType::CPU));

	CHECK_EQ(res.cols(), expected.cols());
	CHECK_EQ(res.rows(), expected.rows());

	for (size_t i = 0; i < res.cols() * res.rows(); i++) {
		CHECK_EQ(res[i / res.rows()][i % res.rows()], expected[i / expected.rows()][i % expected.rows()]);
	}
}

TEST_CASE("cuda_mat_mult BLAS1") {
	tfm::Tensor t(2, 3, tfm::Device(tfm::DeviceType::CPU));
	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		t[i % t.cols()][i / t.cols()] = i;
	}

	tfm::Tensor res = cuda_mat_mult(t, 2.5f);
	t.move_to(tfm::Device(tfm::DeviceType::CPU));
	res.move_to(tfm::Device(tfm::DeviceType::CPU));

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		CHECK_EQ(res[i / res.rows()][i % res.rows()], doctest::Approx(2.5f * t[i / t.rows()][i % t.rows()]));
	}
}

#endif // !NO_CUDA