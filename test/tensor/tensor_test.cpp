#include <doctest.h>
#include <tensor/tensor.h>
#include <compiler_flags.h>


TEST_CASE("Empty tensor") {
	tfm::Tensor empty;
	
	CHECK_EQ(empty.cols(), 0);
	CHECK_EQ(empty.rows(), 0);
	CHECK_EQ(empty.data(), nullptr);
	CHECK_EQ(empty.weights(), nullptr);
	CHECK_EQ(empty.bias(), nullptr);
	CHECK_FALSE(empty.has_bias());
	CHECK_FALSE(empty.has_weights());
}
TEST_CASE("Not empty tensor") {
	tfm::Tensor t(2, 2, tfm::Device(tfm::DeviceType::CPU));
	t[0][0] = 1;
	t[0][1] = 0;
	t[1][0] = 2;
	t[1][1] = 3;

	CHECK_EQ(t.cols(), 2);
	CHECK_EQ(t.rows(), 2);
	CHECK_NE(t.data(), nullptr);
	CHECK_EQ(t.weights(), nullptr);
	CHECK_EQ(t.bias(), nullptr);
	CHECK_FALSE(t.has_bias());
	CHECK_FALSE(t.has_weights());
	CHECK_EQ(t[1][1], 3);
}
TEST_CASE("Weights") {
	tfm::Tensor t(3, 2, tfm::Device(tfm::DeviceType::CPU));
	t.init_weights();

	CHECK_NE(t.weights(), nullptr);
	CHECK_EQ(t.bias(), nullptr);
	CHECK_FALSE(t.has_bias());
	CHECK(t.has_weights());
	CHECK_EQ(t.weights()[1], 1.0f);
}
TEST_CASE("Bias") {
	tfm::Tensor t(3, 2, tfm::Device(tfm::DeviceType::CPU));
	t.init_bias();

	CHECK_EQ(t.weights(), nullptr);
	CHECK_NE(t.bias(), nullptr);
	CHECK(t.has_bias());
	CHECK_FALSE(t.has_weights());
	CHECK_EQ(t.bias()[1], 0.0f);
}
TEST_CASE("Copy constructor and assignment operator") {
	tfm::Tensor orig(3, 2, tfm::Device(tfm::DeviceType::CPU));
	for (size_t i = 0; i < orig.cols() * orig.rows(); i++) {
		orig[i / orig.rows()][i % orig.rows()] = i;
	}
	orig.init_bias();
	tfm::Tensor copy_constructed(orig);
	tfm::Tensor copy_assigned = orig;

	CHECK_NE(orig.data(), copy_constructed.data());
	CHECK_EQ(orig[0][1], copy_constructed[0][1]);
	CHECK_NE(orig.data(), copy_assigned.data());
	CHECK_EQ(orig[0][1], copy_assigned[0][1]);
}
TEST_CASE("Swap constructor and assignment operator") {
	tfm::Tensor orig(3, 2, tfm::Device(tfm::DeviceType::CPU));
	for (size_t i = 0; i < orig.cols() * orig.rows(); i++) {
		orig[i / orig.rows()][i % orig.rows()] = i;
	}
	orig.init_bias();
	tfm::Tensor swap_constructed(std::move(orig));

	CHECK_EQ(orig.data(), nullptr);
	CHECK_EQ(1, swap_constructed[0][1]);

	tfm::Tensor swap_assigned = std::move(swap_constructed);

	CHECK_EQ(swap_constructed.data(), nullptr);
	CHECK_EQ(1, swap_assigned[0][1]);
}
TEST_CASE("Non owning copy") {
	tfm::Tensor orig(3, 2, tfm::Device(tfm::DeviceType::CPU));
	orig.init_bias();
	tfm::Tensor another(orig.non_owning_copy());

	CHECK_EQ(orig.cols(), another.cols());
	CHECK_EQ(orig.rows(), another.rows());
	CHECK_EQ(orig.data(), another.data());
	CHECK_EQ(orig.col_data(1), another.col_data(1));
	CHECK_EQ(orig.bias(), another.bias());
	CHECK_EQ(orig.weights(), another.weights());
	CHECK_EQ(orig.has_bias(), another.has_bias());
	CHECK_EQ(orig.has_weights(), another.has_weights());

	orig[0][0] = 4.0f;
	orig[2][1] = 5.0f;

	another = orig.non_owning_copy({ 2, 1 });

	CHECK_EQ(another.cols(), 2);
	CHECK_EQ(another.rows(), 2);
	CHECK_EQ(another.col_data(0), orig.col_data(2));
	CHECK_EQ(another[0][1], 5.0f);
}
TEST_CASE("Concatenate") {
	tfm::Tensor orig(2, 3, tfm::Device(tfm::DeviceType::CPU));
	for (size_t i = 0; i < orig.cols() * orig.rows(); i++) {
		orig[i / orig.rows()][i % orig.rows()] = i * i;
	}
	tfm::Tensor swapped_cols = orig.non_owning_copy({ 1, 0 });
	tfm::Tensor cat_D0 = tfm::Tensor::concatenate({ orig, swapped_cols }, 0);
	CHECK_EQ(cat_D0.cols(), orig.cols() + swapped_cols.cols());
	CHECK_EQ(cat_D0.rows(), orig.rows());
	CHECK_EQ(cat_D0[0][0], 0);
	CHECK_EQ(cat_D0[0][1], 1);
	CHECK_EQ(cat_D0[0][2], 4);
	CHECK_EQ(cat_D0[1][0], 9);
	CHECK_EQ(cat_D0[1][1], 16);
	CHECK_EQ(cat_D0[1][2], 25);
	CHECK_EQ(cat_D0[2][0], 9);
	CHECK_EQ(cat_D0[2][1], 16);
	CHECK_EQ(cat_D0[2][2], 25);
	CHECK_EQ(cat_D0[3][0], 0);
	CHECK_EQ(cat_D0[3][1], 1);
	CHECK_EQ(cat_D0[3][2], 4);

	tfm::Tensor cat_D1 = tfm::Tensor::concatenate({ swapped_cols, orig }, 1);
	CHECK_EQ(cat_D1.cols(), orig.cols());
	CHECK_EQ(cat_D1.rows(), orig.rows() + swapped_cols.rows());
	CHECK_EQ(cat_D1[0][0], 9);
	CHECK_EQ(cat_D1[0][1], 16);
	CHECK_EQ(cat_D1[0][2], 25);
	CHECK_EQ(cat_D1[0][3], 0);
	CHECK_EQ(cat_D1[0][4], 1);
	CHECK_EQ(cat_D1[0][5], 4);
	CHECK_EQ(cat_D1[1][0], 0);
	CHECK_EQ(cat_D1[1][1], 1);
	CHECK_EQ(cat_D1[1][2], 4);
	CHECK_EQ(cat_D1[1][3], 9);
	CHECK_EQ(cat_D1[1][4], 16);
	CHECK_EQ(cat_D1[1][5], 25);
}
TEST_CASE("Subtensor") {
	tfm::Tensor orig(4, 5, tfm::Device(tfm::DeviceType::CPU));
	for (size_t i = 0; i < orig.cols() * orig.rows(); i++) {
		orig[i / orig.rows()][i % orig.rows()] = i;
	}
	
	tfm::Tensor sub = tfm::Tensor::subtensor(orig, 2, 3, 1, 2);

	CHECK_EQ(sub.cols(), 2);
	CHECK_EQ(sub.rows(), 3);
	CHECK_EQ(sub[0][0], 7);
	CHECK_EQ(sub[0][1], 8);
	CHECK_EQ(sub[0][2], 9);
	CHECK_EQ(sub[1][0], 12);
	CHECK_EQ(sub[1][1], 13);
	CHECK_EQ(sub[1][2], 14);
}
#ifndef NO_CUDA
TEST_CASE("Move to") {
	tfm::Tensor t(3, 2, tfm::Device(tfm::DeviceType::CPU));
	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		t[i / t.rows()][i % t.rows()] = i;
	}
	tfm::Tensor expected = t;

	t.move_to(tfm::Device(tfm::DeviceType::CUDA, 0));

	CHECK_NE(t.data(), nullptr);
	CHECK_NE(t.data(), expected.data());

	t.move_to(tfm::Device(tfm::DeviceType::CPU));

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		CHECK_EQ(t[i / t.rows()][i % t.rows()], expected[i / expected.rows()][i % expected.rows()]);
	}
}
#endif // !NO_CUDA

