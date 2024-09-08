#include <layers/encoder.h>
#include <string>


tfm::Encoder::Encoder(size_t num_layers, size_t num_heads, size_t d_model, size_t d_ff, std::string filename) :
	output_(),
	filename_(filename) {

	for (size_t i = 0; i < num_layers; ++i) {
		layers_.emplace_back(num_heads, d_model, d_ff, filename + "layer" + std::to_string(i));
	}
}


const tfm::Tensor tfm::Encoder::forward(const tfm::Tensor& input) {
	output_ = input;  // copy

	for (auto& layer : layers_) {
		// layer returns a view of its output, which replaces output in-place
		output_ = std::move(layer.forward(output_));
	}
	
	return output();
}

void tfm::Encoder::save() const {
	for (size_t i = 0; i < layers_.size(); i++) {
		layers_[i].save();
	}
}
