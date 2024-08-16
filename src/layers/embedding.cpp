
#include <vector>
#include <cstdint>
#include <cstdlib>

#include <layers/embedding.h>


tfm::Embedding::Embedding(size_t vocab_size, size_t d_model, std::string filename) :
	vocab_size(vocab_size), 
	d_model(d_model),
	embedding_matrix(vocab_size, d_model, tfm::Device(tfm::DeviceType::CPU)),
	output_(),
	filename(filename) {

	if (1 == embedding_matrix.loadFromPath(filename + "embedding_matrix")) {
		// file doesn't exist, generate random matrix
		embedding_matrix.random();
	}
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


void tfm::Embedding::save() const {
	embedding_matrix.saveToPath(filename + "embedding_matrix");
}





