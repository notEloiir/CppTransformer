#pragma once

#include <layers/encoder.h>
#include <layers/decoder.h>
#include <layers/embedding.h>
#include <layers/positional_encoding.h>


namespace tfm
{

class Transformer {
public:
	Transformer(int num_layers, int num_heads, int d_model, int d_ff, int vocab_size, int max_seq_len, std::string model_name);

	const tfm::Tensor forward(const std::vector<uint32_t>& src, const std::vector<uint32_t>& tgt);
	const tfm::Tensor output() const { return output_.nonOwningCopy(); }

	void save() const;

private:
	Encoder encoder;
	Decoder decoder;
	Embedding src_embedding;
	Embedding tgt_embedding;
	PositionalEncoding positional_encoding;
	tfm::Tensor output_;
	std::string model_name;

};

}
