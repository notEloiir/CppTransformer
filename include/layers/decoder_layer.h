#pragma once

#include <layers/multi_head_attention.h>
#include <layers/feed_forward.h>
#include <tensor/tensor.h>


namespace tfm {

class DecoderLayer {
public:
	DecoderLayer(size_t num_heads, size_t d_model, size_t d_ff, std::string filename);

	tfm::Tensor forward(const tfm::Tensor& input, const tfm::Tensor& encoder_output);
	tfm::Tensor output() const { return output_.non_owning_copy(); }

	void save() const;

private:
	MultiHeadAttention self_attention_;
	MultiHeadAttention encoder_decoder_attention_;
	FeedForward feed_forward_;
	size_t d_model_;
	tfm::Tensor output_;
	std::string filename_;
};

}
