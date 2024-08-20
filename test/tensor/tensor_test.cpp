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
	CHECK_FALSE(empty.hasBias());
	CHECK_FALSE(empty.hasWeights());
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
	CHECK_FALSE(t.hasBias());
	CHECK_FALSE(t.hasWeights());
	CHECK_EQ(t[1][1], 3);
}
TEST_CASE("Weights") {
	tfm::Tensor t(3, 2, tfm::Device(tfm::DeviceType::CPU));
	t.initWeights();

	CHECK_NE(t.weights(), nullptr);
	CHECK_EQ(t.bias(), nullptr);
	CHECK_FALSE(t.hasBias());
	CHECK(t.hasWeights());
	CHECK_EQ(t.weights()[1], 1.0f);
}
TEST_CASE("Bias") {
	tfm::Tensor t(3, 2, tfm::Device(tfm::DeviceType::CPU));
	t.initBias();

	CHECK_EQ(t.weights(), nullptr);
	CHECK_NE(t.bias(), nullptr);
	CHECK(t.hasBias());
	CHECK_FALSE(t.hasWeights());
	CHECK_EQ(t.bias()[1], 0.0f);
}
TEST_CASE("Copy constructor and assignment operator") {
	tfm::Tensor orig(3, 2, tfm::Device(tfm::DeviceType::CPU));
	for (size_t i = 0; i < orig.cols() * orig.rows(); i++) {
		orig[i / orig.rows()][i % orig.rows()] = i;
	}
	orig.initBias();
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
	orig.initBias();
	tfm::Tensor swap_constructed(std::move(orig));

	CHECK_EQ(orig.data(), nullptr);
	CHECK_EQ(1, swap_constructed[0][1]);

	tfm::Tensor swap_assigned = std::move(swap_constructed);

	CHECK_EQ(swap_constructed.data(), nullptr);
	CHECK_EQ(1, swap_assigned[0][1]);
}
TEST_CASE("Non owning copy") {
	tfm::Tensor orig(3, 2, tfm::Device(tfm::DeviceType::CPU));
	orig.initBias();
	tfm::Tensor another(orig.nonOwningCopy());

	CHECK_EQ(orig.cols(), another.cols());
	CHECK_EQ(orig.rows(), another.rows());
	CHECK_EQ(orig.data(), another.data());
	CHECK_EQ(orig.colData(1), another.colData(1));
	CHECK_EQ(orig.bias(), another.bias());
	CHECK_EQ(orig.weights(), another.weights());
	CHECK_EQ(orig.hasBias(), another.hasBias());
	CHECK_EQ(orig.hasWeights(), another.hasWeights());

	orig[0][0] = 4.0f;
	orig[2][1] = 5.0f;

	another = orig.nonOwningCopy({ 2, 1 });

	CHECK_EQ(another.cols(), 2);
	CHECK_EQ(another.rows(), 2);
	CHECK_EQ(another.colData(0), orig.colData(2));
	CHECK_EQ(another[0][1], 5.0f);
}
#ifndef NO_CUDA
TEST_CASE("Move to") {
	tfm::Tensor t(3, 2, tfm::Device(tfm::DeviceType::CPU));
	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		t[i / t.rows()][i % t.rows()] = i;
	}
	tfm::Tensor expected = t;

	t.moveTo(tfm::Device(tfm::DeviceType::CUDA, 0));

	CHECK_NE(t.data(), nullptr);
	CHECK_NE(t.data(), expected.data());

	t.moveTo(tfm::Device(tfm::DeviceType::CPU));

	for (size_t i = 0; i < t.cols() * t.rows(); i++) {
		CHECK_EQ(t[i / t.rows()][i % t.rows()], expected[i / expected.rows()][i % expected.rows()]);
	}
}
#endif // !NO_CUDA

