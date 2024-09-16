#include <optimizer/SGD.h>


// Dummy constructor holding parameters
tfm::optimizer::SGD::SGD(float lr) :
	param_(nullptr),
	grad_(nullptr),
	lr_(lr) {}


// Proper constructor
tfm::optimizer::SGD::SGD(tfm::Tensor& param, tfm::Tensor& grad, float lr) :
	param_(&param),
	grad_(&grad),
	lr_(lr) {}


// Create pointer to polymorphed Optimizer object from a dummy object holding parameters and bind it to parameters and gradient
std::unique_ptr<tfm::Optimizer> tfm::optimizer::SGD::bind(tfm::Tensor& param, tfm::Tensor& grad) {
	return std::make_unique<tfm::optimizer::SGD>(param, grad, lr_);
}


void tfm::optimizer::SGD::forward() {
	*param_ = *param_ - (*grad_ * lr_);

	grad_->reset();
}
