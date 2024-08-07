#pragma once

#include <tensor/tensor.h>


class MultiHeadAttention {
public:
	MultiHeadAttention(int num_heads, int d_model);

	// Method to perform multi-head attention
	const tfm::Tensor forward(const tfm::Tensor& queries, const tfm::Tensor& keys, const tfm::Tensor& values);
	const tfm::Tensor output() const { return output_.nonOwningCopy(); }

private:
	int num_heads;
	int d_model;
	tfm::Tensor output_;
};

