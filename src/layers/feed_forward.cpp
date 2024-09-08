#include <layers/feed_forward.h>


tfm::FeedForward::FeedForward(size_t d_model, size_t d_ff, std::string filename) :
	d_model_(d_model),
	d_ff_(d_ff),
	W_0_(d_model, d_ff, tfm::Device(tfm::DeviceType::CPU)),
	W_1_(d_ff, d_model, tfm::Device(tfm::DeviceType::CPU)),
	b_0_(1, d_ff, tfm::Device(tfm::DeviceType::CPU)),
	b_1_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	output_(), 
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
}


const tfm::Tensor tfm::FeedForward::forward(const tfm::Tensor& input) {
	
	tfm::Tensor hidden = W_0_ * input + b_0_;

	hidden.ReLU();

	output_ = W_1_ * hidden + b_1_;

	return output();
}


void tfm::FeedForward::save() const {
	W_0_.save_to_path(filename_ + "W_0_");
	W_1_.save_to_path(filename_ + "W_1_");
	b_0_.save_to_path(filename_ + "b_0_");
	b_1_.save_to_path(filename_ + "b_1_");
}
