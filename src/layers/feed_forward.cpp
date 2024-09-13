#include <layers/feed_forward.h>


tfm::FeedForward::FeedForward(size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer& optimizer) :
	d_model_(d_model),
	d_ff_(d_ff),
	W_0_(d_model, d_ff, tfm::Device(tfm::DeviceType::CPU)),
	W_1_(d_ff, d_model, tfm::Device(tfm::DeviceType::CPU)),
	b_0_(1, d_ff, tfm::Device(tfm::DeviceType::CPU)),
	b_1_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	grad_W_0_(d_model, d_ff, tfm::Device(tfm::DeviceType::CPU)),
	grad_W_1_(d_ff, d_model, tfm::Device(tfm::DeviceType::CPU)),
	grad_b_0_(1, d_ff, tfm::Device(tfm::DeviceType::CPU)),
	grad_b_1_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	optimizer_W_0_(optimizer.bind(W_0_, grad_W_0_)),
	optimizer_W_1_(optimizer.bind(W_1_, grad_W_1_)),
	optimizer_b_0_(optimizer.bind(b_0_, grad_b_0_)),
	optimizer_b_1_(optimizer.bind(b_1_, grad_b_1_)),
	filename_(filename) {

	// try to load file, if doesn't exist, generate random matrix
	if (1 == W_0_.load_from_path(filename + "W_0_")) {
		W_0_.random();
	}
	if (1 == W_1_.load_from_path(filename + "W_1_")) {
		W_1_.random();
	}
	if (1 == b_0_.load_from_path(filename + "b_0_")) {
		b_0_.random();
	}
	if (1 == b_1_.load_from_path(filename + "b_1_")) {
		b_1_.random();
	}
	grad_W_0_.fill(0.0f);
	grad_W_1_.fill(0.0f);
	grad_b_0_.fill(0.0f);
	grad_b_1_.fill(0.0f);
}


tfm::Tensor tfm::FeedForward::forward(const tfm::Tensor& input) {
	input_ = input;
	input_.move_to(tfm::Device(tfm::DeviceType::CPU));

	tfm::Tensor hidden = W_0_ * input + b_0_;

	hidden.ReLU();

	return W_1_ * hidden + b_1_;
}


tfm::Tensor tfm::FeedForward::backward(const tfm::Tensor& grad_output) {
	tfm::Tensor hidden = W_0_ * input_ + b_0_;
	hidden.ReLU();

	grad_W_1_ = grad_output.multiply(hidden, false, true);
	grad_b_1_ = grad_output.sum_along_axis(0);

	tfm::Tensor grad_hidden = grad_output.multiply(W_1_, false, true).multiply_elementwise_ReLU_derivative(hidden);

	grad_W_0_ = grad_hidden.multiply(input_, false, true);
	grad_b_0_ = grad_hidden.sum_along_axis(0);

#ifdef SAVE_VRAM
	input_.move_to(tfm::Device(tfm::DeviceType::CPU));
#endif // SAVE_VRAM
	input_.fill(0.0f);


	return grad_hidden;
}


void tfm::FeedForward::update_parameters() {
	optimizer_W_0_->forward();
	optimizer_W_1_->forward();
	optimizer_b_0_->forward();
	optimizer_b_1_->forward();
}


void tfm::FeedForward::save() const {
	W_0_.save_to_path(filename_ + "W_0_");
	W_1_.save_to_path(filename_ + "W_1_");
	b_0_.save_to_path(filename_ + "b_0_");
	b_1_.save_to_path(filename_ + "b_1_");
}
