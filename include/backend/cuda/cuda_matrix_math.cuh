#pragma once

#include <tensor/tensor.h>


// if B is a vector, will be broadcasted
tfm::Tensor cudaMatAdd(const tfm::Tensor& A, const tfm::Tensor& B);
tfm::Tensor cudaMatMult(const tfm::Tensor& A, const tfm::Tensor& B, bool transposeA=false, bool transposeB=false);
tfm::Tensor cudaMatMultBLAS1(const tfm::Tensor& A, float val);
void cudaNormalizeMatrix(tfm::Tensor& matrix);
void cudaReLU(tfm::Tensor& matrix);
void cudaSoftmax(tfm::Tensor& matrix);
