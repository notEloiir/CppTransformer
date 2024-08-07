#include <layers/encoder.h>


tfm::Encoder::Encoder(int num_layers, int num_heads, int d_model, int d_ff) :
	output_() {

	for (int i = 0; i < num_layers; ++i) {
		layers.emplace_back(num_heads, d_model, d_ff);
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