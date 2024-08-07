#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <tensor/tensor_data.h>


namespace tfm
{

// Column-major 2D tensor
class Tensor : public TensorData {
public:
	using TensorData::TensorData;
	using TensorData::operator=;
	using TensorData::operator[];

	// Copy without ownership
	Tensor nonOwningCopy() const;
	Tensor nonOwningCopy(const std::vector<size_t>& colIds) const;  // RAM-side only (won't copy GPU-side)
	Tensor nonOwningCopy(size_t colOffset, size_t cols) const;

	bool isVector() const { return cols() == 1; }
	void zeroes();
	void ones();
	void diag();
	void random();
	// Normalize Tensor in-place
	void normalize();
	void ReLU();
	Tensor multiply(const Tensor& other, bool transposeThis, bool transposeOther);

	Tensor operator+(const Tensor& other) const;
	Tensor operator*(const Tensor& other) const;
	friend std::ostream& operator<<(std::ostream&, const Tensor&);

private:
	// see TensorData
};

}



