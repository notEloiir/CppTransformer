#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <tensor/device.h>


namespace tfm {

// Column-major 2D tensor
class Tensor {
public:
	// No allocations
	Tensor();
	// Column-major matrix of size [cols, rows]
	Tensor(size_t cols, size_t rows, Device device);
	// Tensor from pointer, does not take ownership
	Tensor(size_t cols, size_t rows, Device device, float* allocated_data);
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
	static Tensor subtensor(const Tensor& other, size_t cols, size_t rows, size_t col_offset, size_t row_offset);

	// Copy without ownership
	Tensor non_owning_copy() const;
	Tensor non_owning_copy(const std::vector<size_t>& col_ids) const;  // RAM-side only (won't copy GPU-side)
	Tensor non_owning_copy(size_t cols, size_t col_offset) const;

	size_t rows() const { return rows_; }
	size_t cols() const { return cols_; }
	float* data() const;
	float* col_data(size_t col) const;
	float* weights() const;
	float* bias() const;
	tfm::Device device() const { return device_; }
	bool has_weights() const { return weights_ != nullptr || weights_cuda_ != nullptr; }
	bool has_bias() const { return bias_ != nullptr || bias_cuda_ != nullptr; }
	bool is_vector() const { return cols() == 1; }

	void init_weights();
	void init_bias();
	// Moving will grant ownership on the new device
	void move_to(Device new_device);
	int save_to_path(const std::string& path) const;
	int load_from_path(const std::string& path, bool load_weights_and_bias = false);

	void fill(float val);
	void random();
	// Normalize Tensor in-place
	void normalize();
	void ReLU();
	void ReLU_derivative();
	void softmax();
	void sq();  // element-wise
	void sqrt();  // element-wise
	Tensor multiply(const Tensor& other, bool transpose_this, bool transpose_other) const;
	Tensor sum_along_axis(size_t axis = 0) const;

	Tensor operator+(const Tensor& other) const;
	Tensor operator-(const Tensor& other) const;
	Tensor operator*(const Tensor& other) const;
	Tensor operator*(float val) const;
	Tensor operator/(const Tensor& other) const;  // element-wise
	Tensor operator/(float val) const;
	float* operator[](size_t col) { return col_data(col); }
	const float* operator[](size_t col) const { return col_data(col); }
	friend std::ostream& operator<<(std::ostream&, const Tensor&);

private:
	size_t cols_;
	size_t rows_;
	float* data_;
	// GPU-side data is always stored continuously
	float* data_cuda_;
	// RAM-side data isn't always stored continuously, in particular (so far, only) for token embedding matrix - to avoid unnecessary copying
	float** data_2D_;
	float* weights_;
	float* weights_cuda_;
	float* bias_;
	float* bias_cuda_;
	bool is_owning_;
	bool is_owning_cuda_;
	bool is_data_continuous_;
	Device device_;

	void allocate();
	void deallocate();
	void set_device();
	void allocate_cuda();
	void deallocate_cuda();
	void cleanup();
};

}



