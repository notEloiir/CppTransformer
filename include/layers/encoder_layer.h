#pragma once

#include <layers/multi_head_attention.h>
#include <layers/feed_forward.h>
#include <tensor/tensor.h>


namespace tfm
{

class EncoderLayer {
public:
	EncoderLayer(size_t num_heads, size_t d_model, size_t d_ff, std::string filename);

	tfm::Tensor forward(const tfm::Tensor& input);
	tfm::Tensor output() const { return output_.nonOwningCopy(); }

	void save() const;

private:
	MultiHeadAttention self_attention;
	FeedForward feed_forward;
	size_t d_model;
	tfm::Tensor output_;
	std::string filename;
};

}


