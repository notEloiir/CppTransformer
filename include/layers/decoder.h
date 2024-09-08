#pragma once

#include <vector>
#include <layers/decoder_layer.h>
#include <tensor/tensor.h>


namespace tfm {

class Decoder {
public:
	Decoder(size_t num_layers, size_t num_heads, size_t d_model, size_t d_ff, std::string filename);

	const tfm::Tensor forward(const tfm::Tensor& input, const tfm::Tensor& encoder_output);
	const tfm::Tensor output() const { return output_.non_owning_copy(); }

	void save() const;

private:
	std::vector<tfm::DecoderLayer> layers_;
	tfm::Tensor output_;
	std::string filename_;
};

}
