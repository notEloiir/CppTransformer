#include <optimizer/optimizer.h>


tfm::Optimizer::Optimizer() :
	type_(OptimizerType::SGD_t),
	lr_(1e-4f),
	momentum_factor_(0.0f),
	velocity_(),
	beta_0_(0.0f),
	beta_1_(0.0f),
	moment_0_(),
	moment_1_() {}


tfm::Optimizer tfm::Optimizer::SGD(float lr) {
	tfm::Optimizer o;
	o.lr_ = lr;

	return o;
}


tfm::Optimizer tfm::Optimizer::SGD_with_momentum(float lr, float momentum_factor) {
	tfm::Optimizer o;
	o.lr_ = lr;
	o.momentum_factor_ = momentum_factor;

	return o;
}


tfm::Optimizer tfm::Optimizer::ADAM(float lr, float beta_0, float beta_1) {
	tfm::Optimizer o;
	o.lr_ = lr;
	o.beta_0_ = beta_0;
	o.beta_1_ = beta_1;

	return o;
}


void tfm::Optimizer::forward(tfm::Tensor& param, tfm::Tensor& gradient) {
	switch (type_) {
	case tfm::Optimizer::SGD_t:
		SGD_forward(param, gradient);
		break;
	case tfm::Optimizer::SGDMomentum_t:
		SGD_momentum_forward(param, gradient);
		break;
	case tfm::Optimizer::ADAM_t:
		ADAM_forward(param, gradient);
		break;
	default:
		break;
	}
}


void tfm::Optimizer::SGD_forward(tfm::Tensor& param, tfm::Tensor& gradient) {
	param = param - (gradient * lr_);
}


void tfm::Optimizer::SGD_momentum_forward(tfm::Tensor& param, tfm::Tensor& gradient) {
	velocity_ = (velocity_ * momentum_factor_) - (gradient * lr_);
	param = param - velocity_;
}


void tfm::Optimizer::ADAM_forward(tfm::Tensor& param, tfm::Tensor& gradient) {
	moment_0_ = (moment_0_ * beta_0_) + (gradient * (1.0f - beta_0_));
	gradient.sq();
	moment_1_ = (moment_1_ * beta_1_) + (gradient * (1.0f - beta_1_));

	tfm::Tensor moment0Dashed = (moment_0_ / (1.0f - beta_0_));
	tfm::Tensor moment1DashedSqrt = (moment_1_ / (1.0f - beta_1_));
	moment1DashedSqrt.sqrt();
	tfm::Tensor eps(1, moment_1_.rows(), moment_1_.device());
	eps.fill(FLT_EPSILON);
	param = param - ((moment0Dashed / (moment1DashedSqrt + eps)) * lr_);
}

