#include <layers/decoder_layer.h>


tfm::DecoderLayer::DecoderLayer(size_t num_heads, size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer& optimizer) :
	self_attention_(num_heads, d_model, filename + "self_attention", optimizer),
	encoder_decoder_attention_(num_heads, d_model, filename + "encoder_decoder_attention", optimizer),
	feed_forward_(d_model, d_ff, filename + "feed_forward", optimizer),
	d_model_(d_model),
	filename_(filename) {}


tfm::Tensor tfm::DecoderLayer::forward(const tfm::Tensor& input, const tfm::Tensor& encoder_output) {
	input_ = input;

	tfm::Tensor self_attention_add_norm = self_attention_.forward(input, input, input);
	self_attention_add_norm += input;
	self_attention_res_ = self_attention_add_norm;  // save normalize input for backpropagation
	self_attention_add_norm.normalize();

	tfm::Tensor cross_attention_add_norm = encoder_decoder_attention_.forward(self_attention_add_norm, encoder_output, encoder_output);
	cross_attention_add_norm += self_attention_add_norm;
	cross_attention_res_ = cross_attention_add_norm;
	cross_attention_add_norm.normalize();

	tfm::Tensor feed_forward_add_norm = feed_forward_.forward(self_attention_add_norm);
	feed_forward_add_norm += self_attention_add_norm;
	feed_forward_res_ = feed_forward_add_norm;
	feed_forward_add_norm.normalize();

	return feed_forward_add_norm;
}


std::pair<tfm::Tensor, tfm::Tensor> tfm::DecoderLayer::backward(const tfm::Tensor& grad_output, const tfm::Tensor& encoder_output) {
	tfm::Tensor grad_input;

	grad_input.normalize_backward(feed_forward_res_);
	// Residual gradient + grad ff
	grad_input += feed_forward_.backward(grad_input);

	grad_input.normalize_backward(cross_attention_res_);
	// Residual gradient + grad cross attention
	grad_input += encoder_decoder_attention_.backward(grad_input, input_, encoder_output, encoder_output);
	const tfm::Tensor& grad_encoder_output = encoder_decoder_attention_.get_grad_K();

	grad_input.normalize_backward(self_attention_res_);
	// Residual gradient + grad self attention
	grad_input += self_attention_.backward(grad_input, input_, input_, input_);

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
