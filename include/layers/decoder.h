#pragma once

#include <vector>
#include <layers/decoder_layer.h>
#include <tensor/tensor.h>
#include <optimizer/optimizer.h>


namespace tfm {

class Decoder {
public:
	Decoder(size_t num_layers, size_t num_heads, size_t d_model, size_t d_ff, std::string filename, tfm::Optimizer& optimizer);

	tfm::Tensor forward(const tfm::Tensor& input, const tfm::Tensor& encoder_output);
	std::pair<tfm::Tensor, tfm::Tensor> backward(const tfm::Tensor& grad_output);
	void update_parameters();

	void save() const;

private:
	std::vector<tfm::DecoderLayer> layers_;
	tfm::Tensor encoder_output_;
	tfm::Tensor grad_input_;
	tfm::Tensor grad_encoder_output_;
	std::string filename_;
};

}
