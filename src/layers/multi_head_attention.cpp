#include <layers/multi_head_attention.h>
#include <compiler_flags.h>


tfm::MultiHeadAttention::MultiHeadAttention(size_t num_heads, size_t d_model, std::string filename, tfm::Optimizer optimizer) :
	num_heads_(num_heads),
	d_model_(d_model),
	d_key_(d_model / num_heads),
	W_q_(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	W_k_(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	W_v_(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	W_o_(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	b_q_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	b_k_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	b_v_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	b_o_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	grad_W_q_(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	grad_W_k_(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	grad_W_v_(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	grad_W_o_(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	grad_b_q_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	grad_b_k_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	grad_b_v_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	grad_b_o_(1, d_model, tfm::Device(tfm::DeviceType::CPU)),
	filename_ (filename),
	optimizer_(optimizer) {
	
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
	if (1 == b_q_.load_from_path(filename + "b_q_")) {
		b_q_.random();
	}
	if (1 == b_k_.load_from_path(filename + "b_k_")) {
		b_k_.random();
	}
	if (1 == b_v_.load_from_path(filename + "b_v_")) {
		b_v_.random();
	}
	if (1 == b_o_.load_from_path(filename + "b_o_")) {
		b_o_.random();
	}
	grad_W_q_.fill(0.0f);
	grad_W_k_.fill(0.0f);
	grad_W_v_.fill(0.0f);
	grad_W_o_.fill(0.0f);
	grad_b_q_.fill(0.0f);
	grad_b_k_.fill(0.0f);
	grad_b_v_.fill(0.0f);
	grad_b_o_.fill(0.0f);
}


tfm::Tensor tfm::MultiHeadAttention::forward(const tfm::Tensor& queries, const tfm::Tensor& keys, const tfm::Tensor& values) {
	tfm::Tensor Q = queries * W_q_ + b_q_;
	tfm::Tensor K = keys * W_k_ + b_k_;
	tfm::Tensor V = values * W_v_ + b_v_;

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
	concatenated_heads_ = tfm::Tensor::concatenate(O_heads, 1);

	return concatenated_heads_ * W_o_ + b_o_;
}


tfm::Tensor tfm::MultiHeadAttention::backward(
	const tfm::Tensor& grad_output, const tfm::Tensor& input_Q, const tfm::Tensor& input_K, const tfm::Tensor& input_V) {

	// Recalculate Q, K, V
	tfm::Tensor Q = input_Q * W_q_ + b_q_;
	tfm::Tensor K = input_K * W_k_ + b_k_;
	tfm::Tensor V = input_V * W_v_ + b_v_;

	// Backprop through output linear
	tfm::Tensor grad_concatenated = grad_output.multiply(W_o_, false, true);
	grad_W_o_ += grad_output.multiply(concatenated_heads_, true, false);
	grad_b_o_ += grad_output.sum_along_axis(0);

	// Clear out data and CUDA allocation (if any) from concatenated_heads_
#ifdef SAVE_VRAM
	concatenated_heads_.move_to(tfm::Device(tfm::DeviceType::CPU));
#endif // SAVE_VRAM
	concatenated_heads_.fill(0.0f);

	// Init grad (not in constructor because no seq_len)
	if (grad_Q_total_.empty()) {
		grad_Q_total_ = tfm::Tensor(input_Q.cols(), input_Q.rows(), tfm::Device(tfm::DeviceType::CPU));
		grad_K_total_ = tfm::Tensor(input_K.cols(), input_K.rows(), tfm::Device(tfm::DeviceType::CPU));
		grad_V_total_ = tfm::Tensor(input_V.cols(), input_V.rows(), tfm::Device(tfm::DeviceType::CPU));
	}
	// Clear out past data
	grad_Q_total_.fill(0.0f);
	grad_K_total_.fill(0.0f);
	grad_V_total_.fill(0.0f);

	// Backprop through attention
	for (size_t i = 0; i < num_heads_; i++) {
		tfm::Tensor grad_head = tfm::Tensor::subtensor(grad_concatenated, grad_concatenated.cols(), d_key_, 0, i * d_key_);

		tfm::Tensor Q_head = tfm::Tensor::subtensor(Q, Q.cols(), d_key_, 0, i * d_key_);
		tfm::Tensor K_head = tfm::Tensor::subtensor(K, K.cols(), d_key_, 0, i * d_key_);
		tfm::Tensor V_head = tfm::Tensor::subtensor(V, V.cols(), d_key_, 0, i * d_key_);
		
		tfm::Tensor grad_Q, grad_K, grad_V;
		std::tie(grad_Q, grad_K, grad_V) = attention_head_backward(grad_head, Q_head, K_head, V_head);

		grad_Q_total_ += grad_Q;
		grad_K_total_ += grad_K;
		grad_V_total_ += grad_V;
	}

	// Calculate gradients
	grad_W_q_ += grad_Q_total_.multiply(input_Q, true, false);
	grad_b_q_ += grad_Q_total_.sum_along_axis(0);

	grad_W_k_ += grad_K_total_.multiply(input_K, true, false);
	grad_b_k_ += grad_K_total_.sum_along_axis(0);

	grad_W_v_ += grad_V_total_.multiply(input_V, true, false);
	grad_b_v_ += grad_V_total_.sum_along_axis(0);
	
	// Return grad for Q, others available through get_grad_K() etc
	return grad_Q_total_;
}


tfm::Tensor tfm::MultiHeadAttention::attention_head(const tfm::Tensor& Q, const tfm::Tensor& K, const tfm::Tensor& V) {
	tfm::Tensor scores = (Q.multiply(K, false, true)) * (1.0f / std::sqrt(static_cast<float>(d_key_)));
	scores.softmax();
	return scores * V;
}


std::tuple<tfm::Tensor, tfm::Tensor, tfm::Tensor> tfm::MultiHeadAttention::attention_head_backward(
	const tfm::Tensor& grad_head, const tfm::Tensor& Q, const tfm::Tensor& K, const tfm::Tensor& V) {

	// Recalculate attention scores
	tfm::Tensor scores = Q.multiply(K, false, true);  // Q * K^T
	float scale = 1.0f / std::sqrt(static_cast<float>(d_key_));
	scores = scores * scale;
	// Attention weights
	scores.softmax();

	tfm::Tensor grad_V = grad_head.multiply(scores, true, false);  // grad_head * scores^T

	tfm::Tensor grad_scores = grad_head.multiply(V, false, false);  // grad_head * V^T
	grad_scores.softmax_backward(scores);

	grad_scores = grad_scores * scale;

	tfm::Tensor grad_Q = grad_scores.multiply(K, false, false);  // grad_scores * K
	tfm::Tensor grad_K = grad_scores.multiply(Q, true, false);   // grad_scores^T * Q

	return std::make_tuple(grad_Q, grad_K, grad_V);
}


void tfm::MultiHeadAttention::update_parameters() {
	// Pass through optimizer
	optimizer_.forward(W_q_, grad_W_q_);
	optimizer_.forward(W_k_, grad_W_k_);
	optimizer_.forward(W_v_, grad_W_v_);
	optimizer_.forward(W_o_, grad_W_o_);
	optimizer_.forward(b_q_, grad_b_q_);
	optimizer_.forward(b_k_, grad_b_k_);
	optimizer_.forward(b_v_, grad_b_v_);
	optimizer_.forward(b_o_, grad_b_o_);

#ifdef SAVE_VRAM
	grad_W_q_.move_to(tfm::Device(tfm::DeviceType::CPU));
	grad_W_k_.move_to(tfm::Device(tfm::DeviceType::CPU));
	grad_W_v_.move_to(tfm::Device(tfm::DeviceType::CPU));
	grad_W_o_.move_to(tfm::Device(tfm::DeviceType::CPU));
	grad_b_q_.move_to(tfm::Device(tfm::DeviceType::CPU));
	grad_b_k_.move_to(tfm::Device(tfm::DeviceType::CPU));
	grad_b_v_.move_to(tfm::Device(tfm::DeviceType::CPU));
	grad_b_o_.move_to(tfm::Device(tfm::DeviceType::CPU));
#endif // SAVE_VRAM

	// Clear out gradient data
	grad_W_q_.fill(0.0f);
	grad_W_k_.fill(0.0f);
	grad_W_v_.fill(0.0f);
	grad_W_o_.fill(0.0f);
	grad_b_q_.fill(0.0f);
	grad_b_k_.fill(0.0f);
	grad_b_v_.fill(0.0f);
	grad_b_o_.fill(0.0f);
}


void tfm::MultiHeadAttention::save() const {
	W_q_.save_to_path(filename_ + "W_q_");
	W_k_.save_to_path(filename_ + "W_k_");
	W_v_.save_to_path(filename_ + "W_v_");
	W_o_.save_to_path(filename_ + "W_o_");
	b_q_.save_to_path(filename_ + "b_q_");
	b_k_.save_to_path(filename_ + "b_k_");
	b_v_.save_to_path(filename_ + "b_v_");
	b_o_.save_to_path(filename_ + "b_o_");
}
