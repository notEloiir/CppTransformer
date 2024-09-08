#include <layers/encoder_layer.h>


tfm::EncoderLayer::EncoderLayer(size_t num_heads, size_t d_model, size_t d_ff, std::string filename) :
	self_attention_(num_heads, d_model, filename + "self_attention_"),
	feed_forward_(d_model, d_ff, filename + "feed forward"),
	d_model_(d_model),
	output_(), 
	filename_(filename) {}

tfm::Tensor tfm::EncoderLayer::forward(const tfm::Tensor& input) {
	// assumes add & norm will be calculated again for backpropagation

	// norm is reused to avoid deallocating and allocating memory again as it's not necessary
	tfm::Tensor norm;

	const tfm::Tensor attention_output = self_attention_.forward(input, input, input);
	norm = attention_output + input;
	norm.normalize();

	const tfm::Tensor feed_forward_output = feed_forward_.forward(norm);
	norm = feed_forward_output + norm;
	norm.normalize();

	output_ = std::move(norm);

	return output();
}


void tfm::EncoderLayer::save() const {
	self_attention_.save();
	feed_forward_.save();
}
