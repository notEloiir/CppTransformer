#pragma once

#include <vector>
#include <layers/encoder_layer.h>
#include <tensor/tensor.h>


namespace tfm
{

class Encoder {
public:
	Encoder(int num_layers, int num_heads, int d_model, int d_ff);

	// Method to perform forward pass through the encoder
	const tfm::Tensor forward(const tfm::Tensor& input);
	const tfm::Tensor output() const { return output_.nonOwningCopy(); }

private:
	std::vector<tfm::EncoderLayer> layers;
	tfm::Tensor output_;
};

}