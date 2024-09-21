#pragma once

#include <tensor/tensor.h>


void cuda_normalize_matrix(tfm::Tensor& matrix, const tfm::Tensor& weights, const tfm::Tensor& bias);
void cuda_normalize_matrix_backward(tfm::Tensor& grad_output, const tfm::Tensor& normalize_input,
	const tfm::Tensor& weights, const tfm::Tensor& bias, tfm::Tensor& grad_weights, tfm::Tensor& grad_bias);

void cuda_ReLU(tfm::Tensor& matrix);
void cuda_ReLU_derivative(tfm::Tensor& matrix);

void cuda_softmax(tfm::Tensor& matrix);
void cuda_softmax_backward(tfm::Tensor& grad, const tfm::Tensor& softmax_output);
