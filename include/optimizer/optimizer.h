#pragma once

#include <tensor/tensor.h>


namespace tfm {

class Optimizer {
public:
	Optimizer();

	static Optimizer SGD(float lr = 1e-3);
	static Optimizer SGD_with_momentum(float lr = 1e-3, float momentum_factor = 1e-2f);
	static Optimizer ADAM(float lr = 1e-3, float beta0 = 0.9f, float beta1 = 0.999f);

	void forward(tfm::Tensor& param, tfm::Tensor& gradient);

private:

	enum OptimizerType {
		SGD_t,
		SGDMomentum_t,
		ADAM_t
	};
	OptimizerType type_;
	float lr_;

	// SGD
	void SGD_forward(tfm::Tensor& param, tfm::Tensor& gradient);

	// SGD with momentum
	float momentum_factor_;
	tfm::Tensor velocity_;
	void SGD_momentum_forward(tfm::Tensor& param, tfm::Tensor& gradient);
	
	// ADAM
	float beta_0_, beta_1_;
	tfm::Tensor moment_0_, moment_1_;
	void ADAM_forward(tfm::Tensor& param, tfm::Tensor& gradient);
};

}