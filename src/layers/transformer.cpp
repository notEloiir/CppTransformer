#include <layers/transformer.h>


tfm::Transformer::Transformer(int num_layers, int num_heads, int d_model, int d_ff, int vocab_size, int max_seq_len) :
	encoder(num_layers, num_heads, d_model, d_ff),
	decoder(num_layers, num_heads, d_model, d_ff),
	src_embedding(vocab_size, d_model),
	tgt_embedding(vocab_size, d_model),
	positional_encoding(max_seq_len, d_model),
	output_() {}


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
