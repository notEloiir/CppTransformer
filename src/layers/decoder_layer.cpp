#include <layers/decoder_layer.h>


tfm::DecoderLayer::DecoderLayer(int num_heads, int d_model, int d_ff) :
	self_attention(num_heads, d_model),
	encoder_decoder_attention(num_heads, d_model),
	feed_forward(d_model, d_ff),
	d_model(d_model),
	output_() {}


tfm::Tensor tfm::DecoderLayer::forward(const tfm::Tensor& input, const tfm::Tensor& encoder_output) {
	// assumes add & norm will be calculated again for backpropagation
	
	// norm is reused to avoid deallocating and allocating memory again as it's not necessary
	tfm::Tensor norm;

	const tfm::Tensor self_attention_output = self_attention.forward(input, input, input);
	norm = self_attention_output + input;
	norm.normalize();

	const tfm::Tensor encoder_decoder_attention_output = encoder_decoder_attention.forward(norm, encoder_output, encoder_output);
	norm = encoder_decoder_attention_output + norm;
	norm.normalize();

	const tfm::Tensor feed_forward_output = feed_forward.forward(norm);
	norm = feed_forward_output + norm;
	norm.normalize();

	output_ = std::move(norm);

	return output();
}