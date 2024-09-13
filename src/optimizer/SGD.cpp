#include <optimizer/SGD.h>


tfm::optimizer::SGD::SGD(float lr) :
	param_(nullptr),
	grad_(nullptr),
	lr_(lr) {}


tfm::optimizer::SGD::SGD(tfm::Tensor& param, tfm::Tensor& grad, float lr = 1e-3) :
	param_(&param),
	grad_(&grad),
	lr_(lr) {}


std::unique_ptr<tfm::Optimizer> tfm::optimizer::SGD::bind(tfm::Tensor& param, tfm::Tensor& grad) {
	return std::make_unique<tfm::optimizer::SGD>(new SGD(param, grad, lr_));
}


void tfm::optimizer::SGD::forward() {
	*param_ = *param_ - (*grad_ * lr_);

	clear_gradient(*grad_);
}
