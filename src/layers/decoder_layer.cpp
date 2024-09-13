#include <layers/decoder_layer.h>


tfm::DecoderLayer::DecoderLayer(size_t num_heads, size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer& optimizer) :
	self_attention_(num_heads, d_model, filename + "self_attention", optimizer),
	encoder_decoder_attention_(num_heads, d_model, filename + "encoder_decoder_attention", optimizer),
	feed_forward_(d_model, d_ff, filename + "feed_forward", optimizer),
	d_model_(d_model),
	filename_(filename) {}


tfm::Tensor tfm::DecoderLayer::forward(const tfm::Tensor& input, const tfm::Tensor& encoder_output) {
	tfm::Tensor add_norm;
	input_ = input;

	const tfm::Tensor self_attention_output = self_attention_.forward(input, input, input);
	add_norm = self_attention_output + input;
	add_norm.normalize();

	const tfm::Tensor encoder_decoder_attention_output = encoder_decoder_attention_.forward(add_norm, encoder_output, encoder_output);
	add_norm = encoder_decoder_attention_output + add_norm;
	add_norm.normalize();

	const tfm::Tensor feed_forward_output = feed_forward_.forward(add_norm);
	add_norm = feed_forward_output + add_norm;
	add_norm.normalize();

	return add_norm;
}


std::pair<tfm::Tensor, tfm::Tensor> tfm::DecoderLayer::backward(const tfm::Tensor& grad_output, const tfm::Tensor& encoder_output) {
	tfm::Tensor grad_input;

	tfm::Tensor grad_feed_forward = grad_output;
	grad_feed_forward.normalize_backward();
	tfm::Tensor grad_feed_forward_output = feed_forward_.backward(grad_feed_forward);
	grad_input = grad_feed_forward_output + grad_feed_forward;

	grad_input.normalize_backward();
	tfm::Tensor grad_encoder_decoder_attention = encoder_decoder_attention_.backward(grad_input, input_, encoder_output, encoder_output);
	const tfm::Tensor& grad_encoder_output = encoder_decoder_attention_.get_grad_K();
	grad_input = grad_encoder_decoder_attention + grad_input;

	grad_input.normalize_backward();
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
