#pragma once

#include <tensor/tensor.h>
#include <optimizer/optimizer.h>


namespace tfm {

class FeedForward {
public:
	FeedForward(size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer optimizer);

	tfm::Tensor forward(const tfm::Tensor& input);
	tfm::Tensor backward(const tfm::Tensor& grad_output);
	void update_parameters();

	void save() const;

private:
	size_t d_model_;
	size_t d_ff_;
	tfm::Tensor W_0_;
	tfm::Tensor W_1_;
	tfm::Tensor b_0_;
	tfm::Tensor b_1_;
	tfm::Tensor grad_W_0_;
	tfm::Tensor grad_W_1_;
	tfm::Tensor grad_b_0_;
	tfm::Tensor grad_b_1_;
	tfm::Tensor input_;
	std::string filename_;
	tfm::Optimizer optimizer_;
};

}



