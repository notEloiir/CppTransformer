#include <layers/transformer.h>
#include <filesystem>


tfm::Transformer::Transformer(
	size_t num_layers, size_t num_heads, size_t d_model, size_t d_ff, size_t vocab_size, size_t max_seq_len, 
	std::string model_name, tfm::Optimizer optimizer
) :
	encoder_(num_layers, num_heads, d_model, d_ff, (std::filesystem::path("models") / model_name / "encoder_").string()),
	decoder_(num_layers, num_heads, d_model, d_ff, (std::filesystem::path("models") / model_name / "decoder_").string()),
	src_embedding_(vocab_size, d_model, (std::filesystem::path("models") / model_name / "embedding").string()),
	tgt_embedding_(vocab_size, d_model, (std::filesystem::path("models") / model_name / "embedding").string()),
	positional_encoding_(max_seq_len, d_model),
	output_(),
	model_name_(model_name),
	optimizer_(optimizer) {}


const tfm::Tensor tfm::Transformer::forward(const std::vector<uint32_t>& src, const std::vector<uint32_t>& tgt) {
	const tfm::Tensor src_embeddings = src_embedding_.forward(src);
	const tfm::Tensor tgt_embeddings = tgt_embedding_.forward(tgt);

	const tfm::Tensor src_pos_embeddings = positional_encoding_.forward(src_embeddings);
	const tfm::Tensor tgt_pos_embeddings = positional_encoding_.forward(tgt_embeddings);

	const tfm::Tensor encoder_output = encoder_.forward(src_pos_embeddings);
	tfm::Tensor decoder_output = decoder_.forward(tgt_pos_embeddings, encoder_output);

	output_ = std::move(decoder_output);

	return output();
}


void tfm::Transformer::backward(const tfm::Tensor& loss_grad) {
	/*
	tfm::Tensor decoder_grad = decoder_.backward(loss_grad);
	tfm::Tensor encoder_grad = encoder_.backward(decoder_grad);

	tfm::Tensor tgt_pos_grad = positional_encoding_.backward(decoder_grad);
	tfm::Tensor src_pos_grad = positional_encoding_.backward(encoder_grad);

	tfm::Tensor tgt_embed_grad = tgt_embedding_.backward(tgt_pos_grad);
	tfm::Tensor src_embed_grad = src_embedding_.backward(src_pos_grad);
	*/
}


void tfm::Transformer::update_parameters() {
	/*
	encoder_.update_parameters();
	decoder_.update_parameters();
	src_embedding_.update_parameters();
	tgt_embedding_.update_parameters();
	positional_encoding_.update_parameters();
	*/
}


void tfm::Transformer::save() const {
	src_embedding_.save();
	tgt_embedding_.save();
	encoder_.save();
	decoder_.save();
}
