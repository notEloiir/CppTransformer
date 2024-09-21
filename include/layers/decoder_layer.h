#pragma once

#include <layers/multi_head_attention.h>
#include <layers/normalization.h>
#include <layers/feed_forward.h>
#include <tensor/tensor.h>


namespace tfm {

class DecoderLayer {
public:
	DecoderLayer(size_t num_heads, size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer& optimizer);

	tfm::Tensor forward(const tfm::Tensor& input, const tfm::Tensor& encoder_output);
	std::pair<tfm::Tensor, tfm::Tensor> backward(const tfm::Tensor& grad_output, const tfm::Tensor& encoder_output);
	void update_parameters();
	const tfm::Tensor& input() { return input_; }

	void save() const;

private:
	MultiHeadAttention self_attention_;
	Normalization self_attention_normalize_;
	MultiHeadAttention encoder_decoder_attention_;
	Normalization encoder_decoder_attention_normalize_;
	FeedForward feed_forward_;
	Normalization feed_forward_normalize_;
	size_t d_model_;

	tfm::Tensor input_;
	std::string filename_;
};

}
