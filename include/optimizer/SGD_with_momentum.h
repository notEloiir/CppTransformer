#pragma once

#include <tensor/tensor.h>
#include <optimizer/optimizer.h>


namespace tfm::optimizer {

class SGD_with_momentum : public tfm::Optimizer {
public:
	// Dummy constructor holding parameters
	SGD_with_momentum(float lr = 1e-3f, float momentum_factor = 1e-2f);
	// Proper constructor
	SGD_with_momentum(tfm::Tensor& parameter, tfm::Tensor& gradient, float lr = 1e-3, float momentum_factor = 1e-2f);
		
	// Create pointer to polymorphed Optimizer object from a dummy object holding parameters and bind it to parameters and gradient
	std::unique_ptr<tfm::Optimizer> bind(tfm::Tensor& parameter, tfm::Tensor& gradient);

	// Pass bound parameters and gradient through optimizer
	void forward();

private:
	float lr_;
	float momentum_factor_;
	tfm::Tensor velocity_;
	tfm::Tensor* param_, * grad_;
};

}
