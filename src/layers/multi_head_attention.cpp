#include <layers/multi_head_attention.h>


tfm::MultiHeadAttention::MultiHeadAttention(int num_heads, int d_model, std::string filename) :
	num_heads(num_heads),
	d_model(d_model),
	output_(),
	filename (filename) {}


const tfm::Tensor tfm::MultiHeadAttention::forward(const tfm::Tensor& queries, const tfm::Tensor& keys, const tfm::Tensor& values) {
	// TODO: implement

	return output();
}


void tfm::MultiHeadAttention::save() const {
	// TODO: implement
}
