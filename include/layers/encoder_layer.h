#pragma once

#include <layers/multi_head_attention.h>
#include <layers/feed_forward.h>
#include <tensor/tensor.h>


namespace tfm {

class EncoderLayer {
public:
	EncoderLayer(size_t num_heads, size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer optimizer);

	tfm::Tensor forward(const tfm::Tensor& input);
	tfm::Tensor backward(const tfm::Tensor& grad_output);
	void update_parameters();

	void save() const;

private:
	MultiHeadAttention self_attention_;
	FeedForward feed_forward_;
	size_t d_model_;

	tfm::Tensor input_;

	std::string filename_;
	tfm::Optimizer optimizer_;
};

}


