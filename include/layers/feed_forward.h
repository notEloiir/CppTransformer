#pragma once

#include <tensor/tensor.h>


namespace tfm {

class FeedForward {
public:
	FeedForward(size_t d_model, size_t d_ff, std::string filename);

	const tfm::Tensor forward(const tfm::Tensor& input);
	const tfm::Tensor output() const { return output_.non_owning_copy(); }

	void save() const;

private:
	size_t d_model_;
	size_t d_ff_;
	tfm::Tensor W_0_;
	tfm::Tensor W_1_;
	tfm::Tensor b_0_;
	tfm::Tensor b_1_;
	tfm::Tensor output_;
	std::string filename_;
};

}



