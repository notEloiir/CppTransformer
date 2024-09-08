#pragma once

#include <tensor/tensor.h>


namespace tfm {

class MultiHeadAttention {
public:
	MultiHeadAttention(size_t num_heads, size_t d_model, std::string filename);

	const tfm::Tensor forward(const tfm::Tensor& queries, const tfm::Tensor& keys, const tfm::Tensor& values);
	const tfm::Tensor output() const { return output_.non_owning_copy(); }

	void save() const;

private:
	size_t num_heads_;
	size_t d_model_;
	size_t d_key_;
	
	tfm::Tensor W_q_; // queries weights
	tfm::Tensor W_k_; // keys weights
	tfm::Tensor W_v_; // values weights
	tfm::Tensor W_o_; // output weights

	tfm::Tensor output_;
	std::string filename_;

	tfm::Tensor attention_head(const tfm::Tensor& Q, const tfm::Tensor& K, const tfm::Tensor& V);
};

}
