
#include <vector>
#include <cstdint>
#include <cstdlib>

#include <layers/embedding.h>


tfm::Embedding::Embedding(size_t vocab_size, size_t d_model, std::string filename) :
	vocab_size_(vocab_size), 
	d_model_(d_model),
	embedding_matrix_(vocab_size, d_model, tfm::Device(tfm::DeviceType::CPU)),
	output_(),
	filename_(filename) {

	if (1 == embedding_matrix_.load_from_path(filename + "embedding_matrix_")) {
		// if file doesn't exist, generate random matrix
		embedding_matrix_.random();
	}
}


const tfm::Tensor tfm::Embedding::forward(const std::vector<uint32_t>& tokens) {
	size_t n = tokens.size();
	std::vector<size_t> cols_to_share(n);
	
	for (size_t i = 0; i < n; i++) {
		cols_to_share[i] = tokens[i];
	}

	output_ = embedding_matrix_.non_owning_copy(cols_to_share);

	return output();
}


void tfm::Embedding::save() const {
	embedding_matrix_.save_to_path(filename_ + "embedding_matrix_");
}





