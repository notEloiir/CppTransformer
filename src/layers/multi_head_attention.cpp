#include <layers/multi_head_attention.h>


tfm::MultiHeadAttention::MultiHeadAttention(size_t num_heads, size_t d_model, std::string filename) :
	num_heads_(num_heads),
	d_model_(d_model),
	d_key_(d_model / num_heads),
	W_q_(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	W_k_(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	W_v_(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	W_o_(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	output_(),
	filename_ (filename) {
	
	// try to load file, if doesn't exist, generate random matrix
	if (1 == W_q_.load_from_path(filename + "W_q_")) {
		W_q_.random();
	}
	if (1 == W_k_.load_from_path(filename + "W_k_")) {
		W_k_.random();
	}
	if (1 == W_v_.load_from_path(filename + "W_v_")) {
		W_v_.random();
	}
	if (1 == W_o_.load_from_path(filename + "W_o_")) {
		W_o_.random();
	}
}


const tfm::Tensor tfm::MultiHeadAttention::forward(const tfm::Tensor& queries, const tfm::Tensor& keys, const tfm::Tensor& values) {
	tfm::Tensor Q = queries * W_q_;
	tfm::Tensor K = keys * W_k_;
	tfm::Tensor V = values * W_q_;

	std::vector<tfm::Tensor> Q_heads(num_heads_);
	std::vector<tfm::Tensor> K_heads(num_heads_);
	std::vector<tfm::Tensor> V_heads(num_heads_);
	std::vector<tfm::Tensor> O_heads(num_heads_);

	for (size_t i = 0; i < num_heads_; i++) {
		Q_heads[i] = tfm::Tensor::subtensor(Q, Q.cols(), d_key_, 0, i * d_key_);
		K_heads[i] = tfm::Tensor::subtensor(K, K.cols(), d_key_, 0, i * d_key_);
		V_heads[i] = tfm::Tensor::subtensor(V, V.cols(), d_key_, 0, i * d_key_);

		O_heads[i] = attention_head(Q_heads[i], K_heads[i], V_heads[i]);
	}

	// Concatenate along dim 1
	output_ = tfm::Tensor::concatenate(O_heads, 1);

	return output();
}


tfm::Tensor tfm::MultiHeadAttention::attention_head(const tfm::Tensor& Q, const tfm::Tensor& K, const tfm::Tensor& V) {
	tfm::Tensor scores = (Q.multiply(K, false, true)) * (1.0f / std::sqrt(static_cast<float>(d_key_)));
	scores.softmax();
	
	return scores * V;
}


void tfm::MultiHeadAttention::save() const {
	W_q_.save_to_path(filename_ + "W_q_");
	W_k_.save_to_path(filename_ + "W_k_");
	W_v_.save_to_path(filename_ + "W_v_");
	W_o_.save_to_path(filename_ + "W_o_");
}
