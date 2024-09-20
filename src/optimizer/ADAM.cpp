#include <optimizer/ADAM.h>

tfm::optimizer::ADAM::ADAM(float lr, float beta_0, float beta_1) :
	param_(nullptr),
	grad_(nullptr),
	lr_(lr),
	beta_0_(beta_0),
	beta_1_(beta_1),
	moment_0_(),
	moment_1_() {}


tfm::optimizer::ADAM::ADAM(tfm::Tensor& param, tfm::Tensor& grad, float lr, float beta_0, float beta_1) :
	param_(&param),
	grad_(&grad),
	lr_(lr),
	beta_0_(beta_0),
	beta_1_(beta_1),
	moment_0_(param.shape().first, param.shape().second, tfm::Device(tfm::DeviceType::CPU)),
	moment_1_(param.shape().first, param.shape().second, tfm::Device(tfm::DeviceType::CPU)) {}


std::unique_ptr<tfm::Optimizer> tfm::optimizer::ADAM::bind(tfm::Tensor& param, tfm::Tensor& grad) {
	return std::make_unique<tfm::optimizer::ADAM>(param, grad, lr_, beta_0_, beta_1_);
}


void tfm::optimizer::ADAM::forward() {
	moment_0_ = (moment_0_ * beta_0_) + (*grad_ * (1.0f - beta_0_));
	grad_->sq();
	moment_1_ = (moment_1_ * beta_1_) + (*grad_ * (1.0f - beta_1_));

	tfm::Tensor moment_0_dashed = (moment_0_ / (1.0f - beta_0_));
	tfm::Tensor moment_1_dashed_sqrt = (moment_1_ / (1.0f - beta_1_));
	moment_1_dashed_sqrt.sqrt();
	tfm::Tensor eps(1, moment_1_.rows(), moment_1_.device());
	eps.fill(FLT_EPSILON);
	*param_ = *param_ - ((moment_0_dashed.divide_elementwise(moment_1_dashed_sqrt + eps)) * lr_);

	grad_->reset();
}
