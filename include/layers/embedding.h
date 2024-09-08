#pragma once

#include <vector>
#include <span>
#include <cstdint>
#include <tensor/tensor.h>


namespace tfm {

class Embedding {
public:
	Embedding(size_t vocab_size, size_t d_model, std::string filename);

	const tfm::Tensor forward(const std::vector<uint32_t>& tokens);
	const tfm::Tensor output() const { return output_.non_owning_copy(); }

	void save() const;

private:
	size_t vocab_size_;
	size_t d_model_;
	tfm::Tensor embedding_matrix_;
	tfm::Tensor output_;
	std::string filename_;
};

}
