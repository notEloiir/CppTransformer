#pragma once

#include <tensor/tensor.h>


// if B is a vector, will be broadcasted
tfm::Tensor cpu_mat_add_BLAS3(const tfm::Tensor& A, const tfm::Tensor& B);
tfm::Tensor cpu_mat_add_along_axis(const tfm::Tensor& A, size_t axis);
tfm::Tensor cpu_mat_sub_BLAS3(const tfm::Tensor& A, const tfm::Tensor& B);
tfm::Tensor cpu_mat_mult_BLAS3(const tfm::Tensor& A, const tfm::Tensor& B, bool transpose_A = false, bool transpose_B = false);
tfm::Tensor cpu_mat_mult_BLAS1(const tfm::Tensor& A, float val);
tfm::Tensor cpu_mat_div_BLAS3(const tfm::Tensor& A, const tfm::Tensor& B);  // element-wise
tfm::Tensor cpu_mat_div_BLAS1(const tfm::Tensor& A, float val);
void cpu_normalize_matrix(tfm::Tensor& matrix);
void cpu_ReLU(tfm::Tensor& matrix);
void cpu_ReLU_derivative(tfm::Tensor& matrix);
void cpu_softmax(tfm::Tensor& matrix);
void cpu_sq(tfm::Tensor& matrix);  // element-wise
void cpu_sqrt(tfm::Tensor& matrix);  // element-wise
