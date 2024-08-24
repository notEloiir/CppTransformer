#include <layers/transformer.h>
#include <filesystem>


tfm::Transformer::Transformer(size_t num_layers, size_t num_heads, size_t d_model, size_t d_ff, size_t vocab_size, size_t max_seq_len, std::string model_name="def") :
	encoder(num_layers, num_heads, d_model, d_ff, (std::filesystem::path("models") / model_name / "encoder").string()),
	decoder(num_layers, num_heads, d_model, d_ff, (std::filesystem::path("models") / model_name / "decoder").string()),
	src_embedding(vocab_size, d_model, (std::filesystem::path("models") / model_name / "embedding").string()),
	tgt_embedding(vocab_size, d_model, (std::filesystem::path("models") / model_name / "embedding").string()),
	positional_encoding(max_seq_len, d_model),
	output_(),
	model_name(model_name) {}


const tfm::Tensor tfm::Transformer::forward(const std::vector<uint32_t>& src, const std::vector<uint32_t>& tgt) {
	// Get embeddings for the source and target sequences
	const tfm::Tensor src_embeddings = src_embedding.forward(src);  // (sequence_length, d_model)
	const tfm::Tensor tgt_embeddings = tgt_embedding.forward(tgt);  // (sequence_length, d_model)

	// Add positional encodings
	const tfm::Tensor src_pos_embeddings = positional_encoding.forward(src_embeddings);
	const tfm::Tensor tgt_pos_embeddings = positional_encoding.forward(tgt_embeddings);

	// Pass through encoder and decoder
	const tfm::Tensor encoder_output = encoder.forward(src_pos_embeddings);
	tfm::Tensor decoder_output = decoder.forward(tgt_pos_embeddings, encoder_output);

	output_ = std::move(decoder_output);

	return output();
}


void tfm::Transformer::save() const {
	src_embedding.save();
	tgt_embedding.save();
	encoder.save();
	decoder.save();
}
