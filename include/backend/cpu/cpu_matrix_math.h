#pragma once

#include <tensor/tensor.h>

// if B is a vector, will be broadcasted
tfm::Tensor cpuMatAdd(const tfm::Tensor& A, const tfm::Tensor& B);
tfm::Tensor cpuMatMult(const tfm::Tensor& A, const tfm::Tensor& B, bool transposeA = false, bool transposeB = false);
void cpuNormalizeMatrix(tfm::Tensor& matrix);
void cpuReLU(tfm::Tensor& matrix);
