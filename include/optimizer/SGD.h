#pragma once

#include <tensor/tensor.h>
#include <optimizer/optimizer.h>


namespace tfm::optimizer {

class SGD : public tfm::Optimizer {
	// Dummy constructor holding parameters
	SGD(float lr = 1e-3);
	// Proper constructor
	SGD(tfm::Tensor& parameter, tfm::Tensor& gradient, float lr = 1e-3);

	// Create pointer to polymorphed Optimizer object from a dummy object holding parameters and bind it to parameters and gradient
	std::unique_ptr<tfm::Optimizer> bind(tfm::Tensor& parameter, tfm::Tensor& gradient);

	// Pass bound parameters and gradient through optimizer
	void forward();

private:
	float lr_;
	tfm::Tensor* param_, * grad_;
};

}
