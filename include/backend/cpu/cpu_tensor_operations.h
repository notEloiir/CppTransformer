#pragma once

#include <tensor/tensor.h>

void cpu_normalize_matrix(tfm::Tensor& matrix, const tfm::Tensor& weights, const tfm::Tensor& bias);
void cpu_normalize_matrix_backward(tfm::Tensor& grad_output, const tfm::Tensor& normalize_input,
	const tfm::Tensor& weights, const tfm::Tensor& bias, tfm::Tensor& grad_weights, tfm::Tensor& grad_bias);

void cpu_ReLU(tfm::Tensor& matrix);
void cpu_ReLU_derivative(tfm::Tensor& matrix);

void cpu_softmax(tfm::Tensor& matrix);
void cpu_softmax_backward(tfm::Tensor& grad, const tfm::Tensor& softmax_output);
