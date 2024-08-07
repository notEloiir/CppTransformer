#pragma once

#include <tensor/tensor.h>


// if B is a vector, will be broadcasted
tfm::Tensor cudaMatAdd(const tfm::Tensor& A, const tfm::Tensor& B);
tfm::Tensor cudaMatMult(const tfm::Tensor& A, const tfm::Tensor& B, bool transposeA=false, bool transposeB=false);
void cudaNormalizeMatrix(tfm::Tensor& matrix);
void cudaReLU(tfm::Tensor& matrix);
