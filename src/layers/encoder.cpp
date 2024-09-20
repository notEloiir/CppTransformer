#include <string>

#include <layers/encoder.h>


tfm::Encoder::Encoder(size_t num_layers, size_t num_heads, size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer& optimizer) :
	filename_(filename) {

	for (size_t i = 0; i < num_layers; ++i) {
		layers_.emplace_back(num_heads, d_model, d_ff, filename + "layer" + std::to_string(i), optimizer);
	}
}


tfm::Tensor tfm::Encoder::forward(const tfm::Tensor& input) {
	tfm::Tensor current = input;
	for (size_t i = 0; i < layers_.size(); i++) {
		current = layers_[i].forward(current);
	}

	return current;
}


tfm::Tensor tfm::Encoder::backward(const tfm::Tensor& grad_output) {
	tfm::Tensor current_grad = grad_output;

	for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; i--) {  // cast to int to avoid underflow
		current_grad = layers_[i].backward(current_grad);
	}

	return current_grad;
}


void tfm::Encoder::update_parameters() {
	for (size_t i = 0; i < layers_.size(); i++) {
		layers_[i].update_parameters();
	}
}


void tfm::Encoder::save() const {
	for (size_t i = 0; i < layers_.size(); i++) {
		layers_[i].save();
	}
}
