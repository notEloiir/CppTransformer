#pragma once

#include <tensor/tensor.h>


namespace tfm{

class FeedForward {
public:
	FeedForward(size_t d_model, size_t d_ff, std::string filename);

	// Method to perform feed-forward computation
	const tfm::Tensor forward(const tfm::Tensor& input);
	const tfm::Tensor output() const { return output_.nonOwningCopy(); }

	void save() const;

private:
	size_t d_model;
	size_t d_ff;
	tfm::Tensor W1;
	tfm::Tensor W2;
	tfm::Tensor b1;
	tfm::Tensor b2;
	tfm::Tensor output_;
	std::string filename;
};

}



