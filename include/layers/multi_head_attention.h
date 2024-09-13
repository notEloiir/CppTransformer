#pragma once

#include <tensor/tensor.h>
#include <optimizer/optimizer.h>


namespace tfm {

class MultiHeadAttention {
public:
	MultiHeadAttention(size_t num_heads, size_t d_model, std::string filename, tfm::Optimizer optimizer);

	tfm::Tensor forward(const tfm::Tensor& queries, const tfm::Tensor& keys, const tfm::Tensor& values);
	tfm::Tensor backward(const tfm::Tensor& grad_output, const tfm::Tensor& input_Q, 
		const tfm::Tensor& input_K, const tfm::Tensor& input_V);
	void update_parameters();
	const tfm::Tensor& get_grad_Q() { return grad_Q_total_; }
	const tfm::Tensor& get_grad_K() { return grad_K_total_; }
	const tfm::Tensor& get_grad_V() { return grad_V_total_; }

	void save() const;

private:
	tfm::Tensor attention_head(const tfm::Tensor& Q, const tfm::Tensor& K, const tfm::Tensor& V);
	std::tuple<tfm::Tensor, tfm::Tensor, tfm::Tensor> attention_head_backward(const tfm::Tensor& grad_head, 
		const tfm::Tensor& Q, const tfm::Tensor& K, const tfm::Tensor& V);

	size_t num_heads_;
	size_t d_model_;
	size_t d_key_;
	
	tfm::Tensor W_q_; // queries weights
	tfm::Tensor W_k_; // keys weights
	tfm::Tensor W_v_; // values weights
	tfm::Tensor W_o_; // output weights
	tfm::Tensor b_q_;
	tfm::Tensor b_k_;
	tfm::Tensor b_v_;
	tfm::Tensor b_o_;

	tfm::Tensor concatenated_heads_;
	tfm::Tensor grad_W_q_;
	tfm::Tensor grad_W_k_;
	tfm::Tensor grad_W_v_;
	tfm::Tensor grad_W_o_;
	tfm::Tensor grad_b_q_;
	tfm::Tensor grad_b_k_;
	tfm::Tensor grad_b_v_;
	tfm::Tensor grad_b_o_;
	tfm::Tensor grad_Q_total_;
	tfm::Tensor grad_K_total_;
	tfm::Tensor grad_V_total_;

	std::string filename_;
	tfm::Optimizer optimizer_;
};

}
