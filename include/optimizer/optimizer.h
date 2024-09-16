#pragma once

#include <memory>

#include <tensor/tensor.h>


namespace tfm {

class Optimizer {
public:
	// Create pointer to polymorphed Optimizer object from a dummy object holding optimizer parameters and bind it to parameters and gradient
	virtual std::unique_ptr<tfm::Optimizer> bind(tfm::Tensor& param, tfm::Tensor& gradient) = 0;
	// Pass bound parameters and gradient through optimizer
	virtual void forward() = 0;
};

}
