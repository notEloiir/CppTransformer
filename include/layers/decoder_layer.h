#pragma once

#include <layers/multi_head_attention.h>
#include <layers/feed_forward.h>
#include <tensor/tensor.h>


namespace tfm
{

class DecoderLayer {
public:
	DecoderLayer(int num_heads, int d_model, int d_ff);

	// Method to perform forward pass through the decoder layer
	tfm::Tensor forward(const tfm::Tensor& input, const tfm::Tensor& encoder_output);
	tfm::Tensor output() const { return output_.nonOwningCopy(); }

private:
	MultiHeadAttention self_attention;
	MultiHeadAttention encoder_decoder_attention;
	FeedForward feed_forward;
	int d_model;
	tfm::Tensor output_;
};

}