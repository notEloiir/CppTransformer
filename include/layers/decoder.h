#pragma once

#include <vector>
#include <layers/decoder_layer.h>
#include <tensor/tensor.h>


namespace tfm
{

class Decoder {
public:
	Decoder(int num_layers, int num_heads, int d_model, int d_ff);

	// Method to perform forward pass through the decoder
	const tfm::Tensor forward(const tfm::Tensor& input, const tfm::Tensor& encoder_output);
	const tfm::Tensor output() const { return output_.nonOwningCopy(); }

private:
	std::vector<tfm::DecoderLayer> layers;
	tfm::Tensor output_;
};

}