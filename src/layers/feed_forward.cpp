#include <layers/feed_forward.h>


tfm::FeedForward::FeedForward(size_t d_model, size_t d_ff, std::string filename) :
	d_model(d_model),
	d_ff(d_ff),
	W1(d_model, d_ff, tfm::Device(tfm::DeviceType::CPU)),
	W2(d_ff, d_model, tfm::Device(tfm::DeviceType::CPU)),
	b1(1, d_ff, tfm::Device(tfm::DeviceType::CPU)),
	b2(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	output_(), 
	filename(filename) {

	// try to load file, if doesn't exist, generate random matrix
	if (1 == W1.loadFromPath(filename + "W1")) {
		W1.random();
	}
	if (1 == W2.loadFromPath(filename + "W2")) {
		W2.random();
	}
	if (1 == b1.loadFromPath(filename + "b1")) {
		b1.random();
	}
	if (1 == b2.loadFromPath(filename + "b2")) {
		b2.random();
	}
}


const tfm::Tensor tfm::FeedForward::forward(const tfm::Tensor& input) {
	
	tfm::Tensor hidden = W1 * input + b1;

	hidden.ReLU();

	output_ = W2 * hidden + b2;

	return output();
}


void tfm::FeedForward::save() const {
	W1.saveToPath(filename + "W1");
	W2.saveToPath(filename + "W2");
	b1.saveToPath(filename + "b1");
	b2.saveToPath(filename + "b2");
}
