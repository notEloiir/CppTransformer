#pragma once

#include <tensor/tensor.h>


// if B is a vector, will be broadcasted
tfm::Tensor cuda_mat_add(const tfm::Tensor& A, const tfm::Tensor& B);
void cuda_mat_add_inplace(tfm::Tensor& A, const tfm::Tensor& B);
tfm::Tensor cuda_mat_add_along_axis(const tfm::Tensor& A, size_t axis);

tfm::Tensor cuda_mat_sub(const tfm::Tensor& A, const tfm::Tensor& B);

tfm::Tensor cuda_mat_mult_elementwise(const tfm::Tensor& A, const tfm::Tensor& B);
tfm::Tensor cuda_mat_mult(const tfm::Tensor& A, float val);
tfm::Tensor cuda_mat_mult(const tfm::Tensor& A, const tfm::Tensor& B, bool transpose_A=false, bool transpose_B=false);

tfm::Tensor cuda_mat_div_elementwise(const tfm::Tensor& A, const tfm::Tensor& B);
tfm::Tensor cuda_mat_div(const tfm::Tensor& A, float val);

void cuda_sq(tfm::Tensor& matrix);  // element-wise
void cuda_sqrt(tfm::Tensor& matrix);  // element-wise
