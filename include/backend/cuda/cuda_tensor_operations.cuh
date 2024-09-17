#pragma once

#include <tensor/tensor.h>


void cuda_normalize_matrix(tfm::Tensor& matrix);
void cuda_normalize_matrix_backward(tfm::Tensor& grad_output, const tfm::Tensor& normalize_input);

void cuda_ReLU(tfm::Tensor& matrix);
void cuda_ReLU_derivative(tfm::Tensor& matrix);

void cuda_softmax(tfm::Tensor& matrix);
void cuda_softmax_backward(tfm::Tensor& grad, const tfm::Tensor& softmax_output);
