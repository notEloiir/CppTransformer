#include <layers/decoder_layer.h>


tfm::DecoderLayer::DecoderLayer(size_t num_heads, size_t d_model, size_t d_ff, std::string filename) :
	self_attention(num_heads, d_model, filename + "self_attention"),
	encoder_decoder_attention(num_heads, d_model, filename + "encoder_decoder_attention"),
	feed_forward(d_model, d_ff, filename + "feed_forward"),
	d_model(d_model),
	output_(),
	filename(filename) {}


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


void tfm::DecoderLayer::save() const {
	self_attention.save();
	encoder_decoder_attention.save();
	feed_forward.save();
}
