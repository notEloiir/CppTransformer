
#include <vector>
#include <cstdint>
#include <cstdlib>

#include <layers/embedding.h>


tfm::Embedding::Embedding(size_t vocab_size, size_t d_model) :
	vocab_size(vocab_size), 
	d_model(d_model),
	embedding_matrix(vocab_size, d_model, tfm::Device(tfm::DeviceType::CPU)),
	output_() {

	// TODO:
	// if file exists
	// load embedding_matrix from that file
	// else
	embedding_matrix.random();
}


const tfm::Tensor tfm::Embedding::forward(const std::vector<uint32_t>& tokens) {
	size_t n = tokens.size();
	std::vector<size_t> colsToShare(n);
	
	for (size_t i = 0; i < n; i++) {
		colsToShare[i] = tokens[i];
	}

	output_ = embedding_matrix.nonOwningCopy(colsToShare);

	return output();
}








