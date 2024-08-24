#pragma once

#include <tensor/tensor.h>


namespace tfm {

class MultiHeadAttention {
public:
	MultiHeadAttention(size_t num_heads, size_t d_model, std::string filename);

	// Method to perform multi-head attention
	const tfm::Tensor forward(const tfm::Tensor& queries, const tfm::Tensor& keys, const tfm::Tensor& values);
	const tfm::Tensor output() const { return output_.nonOwningCopy(); }

	void save() const;

private:
	size_t num_heads;
	size_t d_model;
	


	tfm::Tensor output_;
	std::string filename;
};

}
