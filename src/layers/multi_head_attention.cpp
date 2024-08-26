#include <layers/multi_head_attention.h>


tfm::MultiHeadAttention::MultiHeadAttention(size_t num_heads, size_t d_model, std::string filename) :
	num_heads(num_heads),
	d_model(d_model),
	d_key(d_model / num_heads),
	Wq(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	Wk(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	Wv(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	Wo(d_model, d_model, tfm::Device(tfm::DeviceType::CPU)),
	output_(),
	filename (filename) {
	
	// try to load file, if doesn't exist, generate random matrix
	if (1 == Wq.loadFromPath(filename + "Wq")) {
		Wq.random();
	}
	if (1 == Wk.loadFromPath(filename + "Wk")) {
		Wk.random();
	}
	if (1 == Wv.loadFromPath(filename + "Wv")) {
		Wv.random();
	}
	if (1 == Wo.loadFromPath(filename + "Wo")) {
		Wo.random();
	}
}


const tfm::Tensor tfm::MultiHeadAttention::forward(const tfm::Tensor& queries, const tfm::Tensor& keys, const tfm::Tensor& values) {
	tfm::Tensor Q = queries * Wq;
	tfm::Tensor K = keys * Wk;
	tfm::Tensor V = values * Wq;

	std::vector<tfm::Tensor> Q_heads(num_heads);
	std::vector<tfm::Tensor> K_heads(num_heads);
	std::vector<tfm::Tensor> V_heads(num_heads);
	std::vector<tfm::Tensor> O_heads(num_heads);

	for (size_t i = 0; i < num_heads; i++) {
		Q_heads[i] = tfm::Tensor::subtensor(Q, Q.cols(), d_key, 0, i * d_key);
		K_heads[i] = tfm::Tensor::subtensor(K, K.cols(), d_key, 0, i * d_key);
		V_heads[i] = tfm::Tensor::subtensor(V, V.cols(), d_key, 0, i * d_key);

		O_heads[i] = attention_head(Q_heads[i], K_heads[i], V_heads[i]);
	}

	// Concatenate along dim 1
	output_ = tfm::Tensor::concatenate(O_heads, 1);

	return output();
}


tfm::Tensor tfm::MultiHeadAttention::attention_head(const tfm::Tensor& Q, const tfm::Tensor& K, const tfm::Tensor& V) {
	tfm::Tensor scores = (Q.multiply(K, false, true)) * (1.0 / std::sqrt(static_cast<float>(d_key)));
	scores.softmax();
	
	return scores * V;
}


void tfm::MultiHeadAttention::save() const {
	Wq.saveToPath(filename + "Wq");
	Wk.saveToPath(filename + "Wk");
	Wv.saveToPath(filename + "Wv");
	Wo.saveToPath(filename + "Wo");
}
