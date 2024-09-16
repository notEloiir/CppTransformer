#pragma once

#include <tensor/tensor.h>
#include <optimizer/optimizer.h>


namespace tfm::optimizer {

class ADAM : public tfm::Optimizer {
public:
	// Dummy constructor holding parameters
	ADAM(float lr = 1e-3, float beta0 = 0.9f, float beta1 = 0.999f);
	// Proper constructor
	ADAM(tfm::Tensor& parameter, tfm::Tensor& gradient, float lr = 1e-3, float beta0 = 0.9f, float beta1 = 0.999f);

	// Create pointer to polymorphed Optimizer object from a dummy object holding parameters and bind it to parameters and gradient
	std::unique_ptr<tfm::Optimizer> bind(tfm::Tensor& parameter, tfm::Tensor& gradient);

	// Pass bound parameters and gradient through optimizer
	void forward();

private:
	float lr_;
	float beta_0_, beta_1_;
	tfm::Tensor moment_0_, moment_1_;
	tfm::Tensor* param_, * grad_;
};

}
