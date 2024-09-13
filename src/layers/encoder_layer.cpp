#include <layers/encoder_layer.h>


tfm::EncoderLayer::EncoderLayer(size_t num_heads, size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer optimizer) :
	self_attention_(num_heads, d_model, filename + "self_attention", optimizer),
	feed_forward_(d_model, d_ff, filename + "feed_forward", optimizer),
	d_model_(d_model),
	filename_(filename),
	optimizer_(optimizer) {}


tfm::Tensor tfm::EncoderLayer::forward(const tfm::Tensor& input) {
	tfm::Tensor add_norm;
	input_ = input;

	const tfm::Tensor attention_output = self_attention_.forward(input, input, input);
	add_norm = attention_output + input;
	add_norm.normalize();

	const tfm::Tensor feed_forward_output = feed_forward_.forward(add_norm);
	add_norm = feed_forward_output + add_norm;
	add_norm.normalize();

	return add_norm;
}


tfm::Tensor tfm::EncoderLayer::backward(const tfm::Tensor& grad_output) {
	tfm::Tensor grad_input;

	tfm::Tensor grad_feed_forward = grad_output;
	grad_feed_forward.normalize_backward();
	tfm::Tensor grad_feed_forward_output = feed_forward_.backward(grad_feed_forward);
	grad_input = grad_feed_forward_output + grad_feed_forward;

	grad_input.normalize_backward();
	tfm::Tensor grad_self_attention = self_attention_.backward(grad_input, input_, input_, input_);
	grad_input = grad_self_attention + grad_input;

	return grad_input;
}


void tfm::EncoderLayer::update_parameters() {
	self_attention_.update_parameters();
	feed_forward_.update_parameters();
}


void tfm::EncoderLayer::save() const {
	self_attention_.save();
	feed_forward_.save();
}
