#include <layers/decoder.h>
#include <string>


tfm::Decoder::Decoder(size_t num_layers, size_t num_heads, size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer optimizer) :
	filename_(filename),
	optimizer_(optimizer) {

	for (size_t i = 0; i < num_layers; i++) {
		layers_.emplace_back(num_heads, d_model, d_ff, filename + "layer" + std::to_string(i), optimizer);
	}
}


tfm::Tensor tfm::Decoder::forward(const tfm::Tensor& input, const tfm::Tensor& encoder_output) {
	encoder_output_ = encoder_output;

	tfm::Tensor current = input;
	for (size_t i = 0; i < layers_.size(); i++) {
		current = layers_[i].forward(current, encoder_output);
	}

	return current;
}


std::pair<tfm::Tensor, tfm::Tensor> tfm::Decoder::backward(const tfm::Tensor& grad_output) {
	tfm::Tensor current_grad = grad_output;
	tfm::Tensor grad_encoder_output;

	for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; i--) {  // cast to int to avoid underflow
		tfm::Tensor grad_encoder_output_part;

		// Backprop through the i-th layer
		std::tie(current_grad, grad_encoder_output_part) = layers_[i].backward(current_grad, encoder_output_);

		if (i == layers_.size() - 1) {
			grad_encoder_output = grad_encoder_output_part;
		}
		else {
			grad_encoder_output += grad_encoder_output_part;
		}
	}

	return std::make_pair(current_grad, grad_encoder_output);
}


void tfm::Decoder::update_parameters() {
	for (size_t i = 0; i < layers_.size(); i++) {
		layers_[i].update_parameters();
	}
}


void tfm::Decoder::save() const {
	for (size_t i = 0; i < layers_.size(); i++) {
		layers_[i].save();
	}
}
