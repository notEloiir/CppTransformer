#pragma once

#include <tensor/tensor.h>


// if B is a vector, will be broadcasted
tfm::Tensor cpu_mat_add(const tfm::Tensor& A, const tfm::Tensor& B);
void cpu_mat_add_inplace(tfm::Tensor& A, const tfm::Tensor& B);
tfm::Tensor cpu_mat_add_along_axis(const tfm::Tensor& A, size_t axis);

tfm::Tensor cpu_mat_sub(const tfm::Tensor& A, const tfm::Tensor& B);

tfm::Tensor cpu_mat_mult_elementwise(const tfm::Tensor& A, const tfm::Tensor& B);
tfm::Tensor cpu_mat_mult(const tfm::Tensor& A, float val);
tfm::Tensor cpu_mat_mult(const tfm::Tensor& A, const tfm::Tensor& B, bool transpose_A = false, bool transpose_B = false);

tfm::Tensor cpu_mat_div_elementwise(const tfm::Tensor& A, const tfm::Tensor& B);
tfm::Tensor cpu_mat_div(const tfm::Tensor& A, float val);

void cpu_normalize_matrix(tfm::Tensor& matrix);
void cpu_normalize_matrix_backward(tfm::Tensor& normalize_output, const tfm::Tensor& grad_output);
void cpu_ReLU(tfm::Tensor& matrix);
void cpu_ReLU_derivative(tfm::Tensor& matrix);
void cpu_softmax(tfm::Tensor& matrix);
void cpu_softmax_backward(tfm::Tensor& softmax_output, const tfm::Tensor& grad_output);
void cpu_sq(tfm::Tensor& matrix);  // element-wise
void cpu_sqrt(tfm::Tensor& matrix);  // element-wise
