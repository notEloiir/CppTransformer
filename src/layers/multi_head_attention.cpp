#include <layers/multi_head_attention.h>

MultiHeadAttention::MultiHeadAttention(int num_heads, int d_model) :
	num_heads(num_heads),
	d_model(d_model),
	output_() {}

const tfm::Tensor MultiHeadAttention::forward(const tfm::Tensor& queries, const tfm::Tensor& keys, const tfm::Tensor& values) {
	// TODO: implement

	return output();
}
