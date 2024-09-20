
#include <vector>

#include <layers/positional_encoding.h>


tfm::PositionalEncoding::PositionalEncoding(size_t max_seq_len, size_t d_model) :
	max_seq_len_(max_seq_len), 
	d_model_(d_model),
	positional_encoding_matrix_(max_seq_len, d_model, tfm::Device(tfm::DeviceType::CPU)) {

	// positional encoding formula from the original Transformer paper "Attention is All You Need"
	for (int pos = 0; pos < max_seq_len; pos++) {
		for (int i = 0; i < d_model; i++) {
			if (i % 2 == 0) {
				positional_encoding_matrix_[pos][i] = static_cast<float>(sin(pos / pow(10000.0, static_cast<double>(i) / d_model)));
			}
			else {
				positional_encoding_matrix_[pos][i] = static_cast<float>(cos(pos / pow(10000.0, static_cast<double>(i - 1) / d_model)));
			}
		}
	}
}

tfm::Tensor tfm::PositionalEncoding::forward(const tfm::Tensor& token_embeddings) {
	return token_embeddings + positional_encoding_matrix_;
}
