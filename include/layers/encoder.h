#pragma once

#include <vector>
#include <layers/encoder_layer.h>
#include <tensor/tensor.h>


namespace tfm {

class Encoder {
public:
	Encoder(size_t num_layers, size_t num_heads, size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer optimizer);

	tfm::Tensor forward(const tfm::Tensor& input);
	tfm::Tensor backward(const tfm::Tensor& grad_output);
	void update_parameters();

	void save() const;

private:
	std::vector<tfm::EncoderLayer> layers_;
	std::string filename_;
	tfm::Optimizer optimizer_;
};

}
