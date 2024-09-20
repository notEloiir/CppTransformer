#include <layers/encoder_layer.h>


tfm::EncoderLayer::EncoderLayer(size_t num_heads, size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer& optimizer) :
	self_attention_(num_heads, d_model, filename + "self_attention", optimizer),
	feed_forward_(d_model, d_ff, filename + "feed_forward", optimizer),
	d_model_(d_model),
	filename_(filename) {}


tfm::Tensor tfm::EncoderLayer::forward(const tfm::Tensor& input) {
	input_ = input;

	// Forward through self-attention
	tfm::Tensor self_attention_add_norm = self_attention_.forward(input, input, input);
	// Residual addition
	self_attention_add_norm += input;
	// Save input to normalize for backpropagation
	self_attention_res_ = self_attention_add_norm;
	// Normalize
	self_attention_add_norm.normalize();

	tfm::Tensor feed_forward_add_norm = feed_forward_.forward(self_attention_add_norm);
	feed_forward_add_norm += self_attention_add_norm;
	feed_forward_res_ = feed_forward_add_norm;
	feed_forward_add_norm.normalize();

	return feed_forward_add_norm;
}


tfm::Tensor tfm::EncoderLayer::backward(const tfm::Tensor& grad_output) {
	tfm::Tensor grad_input = grad_output;

	grad_input.normalize_backward(feed_forward_res_);
	// Residual gradient + grad ff
	grad_input += feed_forward_.backward(grad_input);

	grad_input.normalize_backward(self_attention_res_);
	// Residual gradient + grad self attention
	grad_input += self_attention_.backward(grad_input, input_, input_, input_);

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
