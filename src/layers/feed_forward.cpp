#include <layers/feed_forward.h>

FeedForward::FeedForward(int d_model, int d_ff) :
	d_model(d_model),
	d_ff(d_ff),
	W1(d_model, d_ff, tfm::Device(tfm::DeviceType::CPU)),
	W2(d_ff, d_model, tfm::Device(tfm::DeviceType::CPU)),
	b1(1, d_ff, tfm::Device(tfm::DeviceType::CPU)),
	b2(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	output_() {}

const tfm::Tensor FeedForward::forward(const tfm::Tensor& input) {
	
	tfm::Tensor hidden = W1 * input + b1;

	hidden.ReLU();

	output_ = W2 * hidden + b2;

	return output();
}
