
#include <vector>
#include <cstdint>
#include <cstdlib>

#include <layers/embedding.h>
#include <compiler_flags.h>


tfm::Embedding::Embedding(size_t vocab_size, size_t d_model, std::string filename, tfm::Optimizer optimizer) :
	vocab_size_(vocab_size), 
	d_model_(d_model),
	embedding_matrix_(vocab_size, d_model, tfm::Device(tfm::DeviceType::CPU)),
	grad_(vocab_size, d_model, tfm::Device(tfm::DeviceType::CPU)),
	filename_(filename), 
	optimizer_(optimizer) {

	if (1 == embedding_matrix_.load_from_path(filename + "embedding_matrix_")) {
		// if file doesn't exist, generate random matrix
		embedding_matrix_.random();
	}
	grad_.fill(0.0f);
}


tfm::Tensor tfm::Embedding::forward(const std::vector<uint32_t>& tokens) {
	size_t n = tokens.size();
	std::vector<size_t> cols_to_share(n);
	input_token_indices_ = cols_to_share;
	
	for (size_t i = 0; i < n; i++) {
		cols_to_share[i] = tokens[i];
	}

	return embedding_matrix_.non_owning_copy(cols_to_share);
}


tfm::Tensor tfm::Embedding::backward(const tfm::Tensor& grad_output) {
	for (size_t i = 0; i < input_token_indices_.size(); ++i) {
		float* grad_token = grad_output.col_data(i);
		grad_.add_to_col(input_token_indices_[i], grad_token);
	}

	return grad_;
}


void tfm::Embedding::update_parameters() {
	optimizer_.forward(embedding_matrix_, grad_);

#ifdef SAVE_VRAM
	grad_.move_to(tfm::Device(tfm::DeviceType::CPU));
#endif // SAVE_VRAM
	grad_.fill(0.0f);
}


void tfm::Embedding::save() const {
	embedding_matrix_.save_to_path(filename_ + "embedding_matrix_");
}





