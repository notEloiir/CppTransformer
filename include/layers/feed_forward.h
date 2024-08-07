#pragma once

#include <tensor/tensor.h>


class FeedForward {
public:
	FeedForward(int d_model, int d_ff);

	// Method to perform feed-forward computation
	const tfm::Tensor forward(const tfm::Tensor& input);
	const tfm::Tensor output() const { return output_.nonOwningCopy(); }
private:
	int d_model;
	int d_ff;
	tfm::Tensor W1;
	tfm::Tensor W2;
	tfm::Tensor b1;
	tfm::Tensor b2;
	tfm::Tensor output_;
};





