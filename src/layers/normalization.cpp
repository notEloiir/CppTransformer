#include <layers/normalization.h>


tfm::Normalization::Normalization(size_t d_model, std::string filename, tfm::Optimizer& optimizer) :
	d_model_(d_model),
	weights_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	bias_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	grad_weights_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	grad_bias_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	optimizer_weights_(optimizer.bind(weights_, grad_weights_)),
	optimizer_bias_(optimizer.bind(bias_, grad_bias_)) {

	// Load weights and bias if exist
	if (1 == weights_.load_from_path(filename + "weights_")) {
		weights_.random();
	}
	if (1 == bias_.load_from_path(filename + "bias_")) {
		bias_.random();
	}

	// Initialize gradient
	grad_weights_.fill(0.0f);
	grad_bias_.fill(0.0f);
}


void tfm::Normalization::forward(tfm::Tensor& input) {
	input_ = input;
	input.normalize(weights_, bias_);
}


void tfm::Normalization::backward(tfm::Tensor& grad_output) {
	grad_output.normalize_backward(input_, weights_, bias_, grad_weights_, grad_bias_);
	input_.reset();
}


void tfm::Normalization::update_parameters() {
	optimizer_weights_->forward();
	optimizer_bias_->forward();
}


void tfm::Normalization::save() const {
	weights_.save_to_path(filename_ + "weights_");
	bias_.save_to_path(filename_ + "bias_");
}
