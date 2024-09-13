#pragma once

#include <layers/encoder.h>
#include <layers/decoder.h>
#include <layers/embedding.h>
#include <layers/positional_encoding.h>
#include <optimizer/optimizer.h>


namespace tfm {

class Transformer {
public:
	Transformer(
		size_t num_layers, size_t num_heads, size_t d_model, size_t d_ff, size_t vocab_size, size_t max_seq_len, 
		std::string model_name, tfm::Optimizer optimizer
	);

	tfm::Tensor forward(const std::vector<uint32_t>& src, const std::vector<uint32_t>& tgt);
	void backward(const tfm::Tensor& loss_grad);
	void update_parameters();

	void save() const;

private:
	Encoder encoder_;
	Decoder decoder_;
	Embedding src_embedding_;
	Embedding tgt_embedding_;
	PositionalEncoding positional_encoding_;
	std::string model_name_;
	tfm::Optimizer optimizer_;
};

}
