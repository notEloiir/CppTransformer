#pragma once

#include <vector>
#include <span>
#include <cstdint>
#include <tensor/tensor.h>


namespace tfm
{

class Embedding {
public:
	Embedding(size_t vocab_size, size_t d_model);

	// Convert tokens to their embeddings
	const tfm::Tensor forward(const std::vector<uint32_t>& tokens);
	const tfm::Tensor output() const { return output_.nonOwningCopy(); }

private:
	size_t vocab_size;
	size_t d_model;
	tfm::Tensor embedding_matrix;
	tfm::Tensor output_;
};


}
