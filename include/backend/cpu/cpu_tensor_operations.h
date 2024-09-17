#pragma once

#include <tensor/tensor.h>

void cpu_normalize_matrix(tfm::Tensor& matrix);
void cpu_normalize_matrix_backward(tfm::Tensor& grad_output, const tfm::Tensor& normalize_input);

void cpu_ReLU(tfm::Tensor& matrix);
void cpu_ReLU_derivative(tfm::Tensor& matrix);

void cpu_softmax(tfm::Tensor& matrix);
void cpu_softmax_backward(tfm::Tensor& grad, const tfm::Tensor& softmax_output);
