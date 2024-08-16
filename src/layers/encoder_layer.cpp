#include <layers/encoder_layer.h>


tfm::EncoderLayer::EncoderLayer(int num_heads, int d_model, int d_ff, std::string filename) :
	self_attention(num_heads, d_model, filename + "self_attention"),
	feed_forward(d_model, d_ff, filename + "feed forward"),
	d_model(d_model),
	output_(), 
	filename(filename) {}

tfm::Tensor tfm::EncoderLayer::forward(const tfm::Tensor& input) {
	// assumes add & norm will be calculated again for backpropagation

	// norm is reused to avoid deallocating and allocating memory again as it's not necessary
	tfm::Tensor norm;

	const tfm::Tensor attention_output = self_attention.forward(input, input, input);
	norm = attention_output + input;
	norm.normalize();

	const tfm::Tensor feed_forward_output = feed_forward.forward(norm);
	norm = feed_forward_output + norm;
	norm.normalize();

	output_ = std::move(norm);

	return output();
}


void tfm::EncoderLayer::save() const {
	self_attention.save();
	feed_forward.save();
}
