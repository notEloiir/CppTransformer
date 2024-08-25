#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <tensor/device.h>


namespace tfm
{

// Column-major 2D tensor
class Tensor {
public:
	// No allocations
	Tensor();
	// Column-major matrix of size [cols, rows]
	Tensor(size_t cols, size_t rows, Device device);
	// Copy (deep, has ownership)
	Tensor(const Tensor& other);
	Tensor& operator=(const Tensor& other);
	// Move
	Tensor(Tensor&& other) noexcept;
	Tensor& operator=(Tensor&& other) noexcept;
	// Destructor
	~Tensor() { cleanup(); }

	// Concatenate tensors
	static Tensor concatenate(const std::vector<tfm::Tensor>& tensors, size_t dim = 0);
	// Submatrix of size [cols, rows] and given offsets (deep copy, has ownership)
	static Tensor subtensor(const Tensor& other, size_t cols, size_t rows, size_t colOffset, size_t rowOffset);

	// Copy without ownership
	Tensor nonOwningCopy() const;
	Tensor nonOwningCopy(const std::vector<size_t>& colIds) const;  // RAM-side only (won't copy GPU-side)
	Tensor nonOwningCopy(size_t cols, size_t colOffset) const;

	size_t rows() const { return rows_; }
	size_t cols() const { return cols_; }
	float* data() const;
	float* colData(size_t col) const;
	float* weights() const;
	float* bias() const;
	tfm::Device device() const { return device_; }
	bool hasWeights() const { return weights_ != nullptr || weightsCuda_ != nullptr; }
	bool hasBias() const { return bias_ != nullptr || biasCuda_ != nullptr; }
	bool isVector() const { return cols() == 1; }

	void initWeights();
	void initBias();
	// Moving will grant ownership on the new device
	void moveTo(Device newDevice);
	int saveToPath(const std::string& path) const;
	int loadFromPath(const std::string& path, bool loadWeightsAndBias = false);

	void zeroes();
	void ones();
	void diag();
	void random();
	// Normalize Tensor in-place
	void normalize();
	void ReLU();
	void softmax();
	Tensor multiply(const Tensor& other, bool transposeThis, bool transposeOther) const;

	Tensor operator+(const Tensor& other) const;
	Tensor operator*(const Tensor& other) const;
	Tensor operator*(float val) const;
	float* operator[](size_t col) { return colData(col); }
	const float* operator[](size_t col) const { return colData(col); }
	friend std::ostream& operator<<(std::ostream&, const Tensor&);

private:
	size_t cols_;
	size_t rows_;
	float* data_;
	// GPU-side data is always stored continuously
	float* dataCuda_;
	// RAM-side data isn't always stored continuously, in particular (so far, only) for token embedding matrix - to avoid unnecessary copying
	float** data2D_;
	float* weights_;
	float* weightsCuda_;
	float* bias_;
	float* biasCuda_;
	bool isOwning_;
	bool isOwningCuda_;
	bool isDataContinuous_;
	Device device_;

	void allocate();
	void deallocate();
	void setDevice();
	void allocateCuda();
	void deallocateCuda();
	void cleanup();
};

}



