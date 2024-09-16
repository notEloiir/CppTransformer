#include <layers/decoder_layer.h>


tfm::DecoderLayer::DecoderLayer(size_t num_heads, size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer& optimizer) :
	self_attention_(num_heads, d_model, filename + "self_attention", optimizer),
	encoder_decoder_attention_(num_heads, d_model, filename + "encoder_decoder_attention", optimizer),
	feed_forward_(d_model, d_ff, filename + "feed_forward", optimizer),
	d_model_(d_model),
	filename_(filename) {}


tfm::Tensor tfm::DecoderLayer::forward(const tfm::Tensor& input, const tfm::Tensor& encoder_output) {
	input_ = input;

	self_attention_add_norm_ = self_attention_.forward(input, input, input);
	self_attention_add_norm_ += input;
	self_attention_add_norm_.normalize();

	cross_attention_add_norm_ = encoder_decoder_attention_.forward(self_attention_add_norm_, encoder_output, encoder_output);
	cross_attention_add_norm_ += self_attention_add_norm_;
	cross_attention_add_norm_.normalize();

	feed_forward_add_norm_ = feed_forward_.forward(cross_attention_add_norm_);
	feed_forward_add_norm_ += cross_attention_add_norm_;
	feed_forward_add_norm_.normalize();

	return feed_forward_add_norm_;
}


std::pair<tfm::Tensor, tfm::Tensor> tfm::DecoderLayer::backward(const tfm::Tensor& grad_output, const tfm::Tensor& encoder_output) {
	tfm::Tensor grad_input;

	tfm::Tensor grad_feed_forward = grad_output;
	feed_forward_add_norm_.normalize_backward(grad_feed_forward);
	tfm::Tensor grad_feed_forward_output = feed_forward_.backward(grad_feed_forward);
	grad_input = grad_feed_forward_output + grad_feed_forward;

	cross_attention_add_norm_.normalize_backward(grad_input);
	tfm::Tensor grad_encoder_decoder_attention = encoder_decoder_attention_.backward(grad_input, input_, encoder_output, encoder_output);
	const tfm::Tensor& grad_encoder_output = encoder_decoder_attention_.get_grad_K();
	grad_input = grad_encoder_decoder_attention + grad_input;

	self_attention_add_norm_.normalize_backward(grad_input);
	tfm::Tensor grad_self_attention = self_attention_.backward(grad_input, input_, input_, input_);
	grad_input = grad_self_attention + grad_input;

	return std::make_pair(grad_input, grad_encoder_output);
}


void tfm::DecoderLayer::update_parameters() {
	self_attention_.update_parameters();
	encoder_decoder_attention_.update_parameters();
	feed_forward_.update_parameters();
}


void tfm::DecoderLayer::save() const {
	self_attention_.save();
	encoder_decoder_attention_.save();
	feed_forward_.save();
}
