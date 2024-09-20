#pragma once

#include <tensor/tensor.h>


void cuda_fill(float* dev_ptr, float val, size_t cols, size_t rows);
