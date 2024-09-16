#include <optimizer/SGD_with_momentum.h>


tfm::optimizer::SGD_with_momentum::SGD_with_momentum(float lr, float momentum_factor) :
	param_(nullptr),
	grad_(nullptr),
	lr_(lr),
	momentum_factor_(momentum_factor),
	velocity_() {}


tfm::optimizer::SGD_with_momentum::SGD_with_momentum(tfm::Tensor& param, tfm::Tensor& grad, float lr, float momentum_factor) :
	param_(&param),
	grad_(&grad),
	lr_(lr),
	momentum_factor_(momentum_factor),
	velocity_(param.shape().first, param.shape().second, tfm::Device(tfm::DeviceType::CPU)) {}


std::unique_ptr<tfm::Optimizer> tfm::optimizer::SGD_with_momentum::bind(tfm::Tensor& param, tfm::Tensor& grad) {
	return std::make_unique<tfm::optimizer::SGD_with_momentum>(param, grad, lr_, momentum_factor_);
}


void tfm::optimizer::SGD_with_momentum::forward() {
	velocity_ = (velocity_ * momentum_factor_) - (*grad_ * lr_);
	*param_ = *param_ - velocity_;

	grad_->reset();
}
