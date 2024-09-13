#pragma once

#include <vector>
#include <span>
#include <cstdint>
#include <tensor/tensor.h>
#include <optimizer/optimizer.h>


namespace tfm {

class Embedding {
public:
	Embedding(size_t vocab_size, size_t d_model, std::string filename, tfm::Optimizer optimizer);

	tfm::Tensor forward(const std::vector<uint32_t>& tokens);
	tfm::Tensor backward(const tfm::Tensor& grad_output);
	void update_parameters();

	void save() const;

private:
	size_t vocab_size_;
	size_t d_model_;
	tfm::Tensor embedding_matrix_;
	tfm::Tensor grad_;
	std::vector<size_t> input_token_indices_;
	std::string filename_;
	tfm::Optimizer optimizer_;
};

}
