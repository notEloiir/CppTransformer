#include <layers/encoder.h>
#include <string>


tfm::Encoder::Encoder(int num_layers, int num_heads, int d_model, int d_ff, std::string filename) :
	output_(),
	filename(filename) {

	for (int i = 0; i < num_layers; ++i) {
		layers.emplace_back(num_heads, d_model, d_ff, filename + "layer" + std::to_string(i));
	}
}


const tfm::Tensor tfm::Encoder::forward(const tfm::Tensor& input) {
	output_ = input;  // copy

	for (auto& layer : layers) {
		// layer returns a view of its output, which replaces output in-place
		output_ = std::move(layer.forward(output_));
	}
	
	return output();
}

void tfm::Encoder::save() const {
	for (int i = 0; i < layers.size(); i++) {
		layers[i].save();
	}
}
