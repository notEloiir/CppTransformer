#pragma once

#include <vector>
#include <cstdint>
#include <tensor/tensor.h>


namespace tfm
{

	class PositionalEncoding {
	public:
		PositionalEncoding(size_t max_seq_len, size_t d_model);

		tfm::Tensor forward(const tfm::Tensor& token_embeddings);

	private:
		size_t max_seq_len;
		size_t d_model;
		tfm::Tensor positional_encoding_matrix;
	};


}
