#include <layers/encoder_layer.h>


tfm::EncoderLayer::EncoderLayer(size_t num_heads, size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer& optimizer) :
	self_attention_(num_heads, d_model, filename + "self_attention", optimizer),
	feed_forward_(d_model, d_ff, filename + "feed_forward", optimizer),
	d_model_(d_model),
	filename_(filename) {}


tfm::Tensor tfm::EncoderLayer::forward(const tfm::Tensor& input) {
	input_ = input;

	self_attention_add_norm_ = self_attention_.forward(input, input, input);
	self_attention_add_norm_ += input;
	self_attention_add_norm_.normalize();

	feed_forward_add_norm_ = feed_forward_.forward(self_attention_add_norm_);
	feed_forward_add_norm_ += self_attention_add_norm_;
	feed_forward_add_norm_.normalize();

	return feed_forward_add_norm_;
}


tfm::Tensor tfm::EncoderLayer::backward(const tfm::Tensor& grad_output) {
	tfm::Tensor grad_input = grad_output;

	feed_forward_add_norm_.normalize_backward(grad_input);
	tfm::Tensor grad_feed_forward_output = feed_forward_.backward(feed_forward_add_norm_);
	grad_input += grad_feed_forward_output;

	self_attention_add_norm_.normalize_backward(grad_input);
	tfm::Tensor grad_self_attention = self_attention_.backward(self_attention_add_norm_, input_, input_, input_);
	grad_input += grad_self_attention;

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
