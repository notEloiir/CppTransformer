#include <layers/decoder.h>
#include <string>


tfm::Decoder::Decoder(size_t num_layers, size_t num_heads, size_t d_model, size_t d_ff, std::string filename) :
	output_(),
	filename(filename) {

	for (size_t i = 0; i < num_layers; i++) {
		layers.emplace_back(num_heads, d_model, d_ff, filename + "layer" + std::to_string(i));
	}
}


const tfm::Tensor tfm::Decoder::forward(const tfm::Tensor& input, const tfm::Tensor& encoder_output) {
	output_ = input;  // copy

	for (auto& layer : layers) {
		output_ = std::move(layer.forward(output_, encoder_output));
	}

	return output();
}


void tfm::Decoder::save() const {
	for (size_t i = 0; i < layers.size(); i++) {
		layers[i].save();
	}
}
