#include <layers/encoder_layer.h>


tfm::EncoderLayer::EncoderLayer(size_t num_heads, size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer& optimizer) :
	self_attention_(num_heads, d_model, filename + "self_attention", optimizer),
	feed_forward_(d_model, d_ff, filename + "feed_forward", optimizer),
	self_attention_normalize_(d_model, filename + "self_att_normalize", optimizer),
	feed_forward_normalize_(d_model, filename + "ff_normalize", optimizer),
	d_model_(d_model),
	filename_(filename) {}


tfm::Tensor tfm::EncoderLayer::forward(const tfm::Tensor& input) {
	input_ = input;

	// Forward through self-attention
	tfm::Tensor self_attention_add_norm = self_attention_.forward(input, input, input);
	// Residual addition
	self_attention_add_norm += input;
	// Normalize the result
	self_attention_normalize_.forward(self_attention_add_norm);

	tfm::Tensor feed_forward_add_norm = feed_forward_.forward(self_attention_add_norm);
	feed_forward_add_norm += self_attention_add_norm;
	feed_forward_normalize_.forward(feed_forward_add_norm);

	return feed_forward_add_norm;
}


tfm::Tensor tfm::EncoderLayer::backward(const tfm::Tensor& grad_output) {
	tfm::Tensor grad_input = grad_output;

	feed_forward_normalize_.backward(grad_input);
	// Residual gradient + grad ff
	grad_input += feed_forward_.backward(grad_input);

	self_attention_normalize_.backward(grad_input);
	// Residual gradient + grad self attention
	grad_input += self_attention_.backward(grad_input, input_, input_, input_);

	return grad_input;
}


void tfm::EncoderLayer::update_parameters() {
	self_attention_.update_parameters();
	self_attention_normalize_.update_parameters();
	feed_forward_.update_parameters();
	feed_forward_normalize_.update_parameters();
}


void tfm::EncoderLayer::save() const {
	self_attention_.save();
	self_attention_normalize_.save();
	feed_forward_.save();
	feed_forward_normalize_.save();
}
