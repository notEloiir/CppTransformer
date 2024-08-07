#include <layers/encoder_layer.h>


tfm::EncoderLayer::EncoderLayer(int num_heads, int d_model, int d_ff) :
	self_attention(num_heads, d_model), 
	feed_forward(d_model, d_ff), 
	d_model(d_model),
	output_() {}

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

