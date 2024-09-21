#pragma once

#include <tensor/tensor.h>
#include <optimizer/optimizer.h>


namespace tfm {

class Normalization {
public:
	Normalization(size_t d_model, std::string filename, tfm::Optimizer& optimizer);

	void forward(tfm::Tensor& input);
	void backward(tfm::Tensor& grad_output);
	void update_parameters();

	void save() const;

private:
	size_t d_model_;

	tfm::Tensor input_;
	tfm::Tensor weights_;
	tfm::Tensor bias_;
	tfm::Tensor grad_weights_;
	tfm::Tensor grad_bias_;
	std::unique_ptr<tfm::Optimizer> optimizer_weights_;
	std::unique_ptr<tfm::Optimizer> optimizer_bias_;

	std::string filename_;
};

}
