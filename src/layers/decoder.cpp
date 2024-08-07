#include <layers/decoder.h>


tfm::Decoder::Decoder(int num_layers, int num_heads, int d_model, int d_ff) :
	output_() {

	for (int i = 0; i < num_layers; ++i) {
		layers.emplace_back(num_heads, d_model, d_ff);
	}
}


const tfm::Tensor tfm::Decoder::forward(const tfm::Tensor& input, const tfm::Tensor& encoder_output) {
	output_ = input;  // copy

	for (auto& layer : layers) {
		output_ = std::move(layer.forward(output_, encoder_output));
	}

	return output();
}

