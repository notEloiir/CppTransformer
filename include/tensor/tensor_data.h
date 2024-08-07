#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>
#include <tensor/device.h>


namespace tfm
{

	// Column-major 2D matrix data manager
	class TensorData {
	public:
		// No allocations
		TensorData();
		// Column-major matrix of size [cols, rows]
		TensorData(size_t cols, size_t rows, Device device);
		// Copy (deep, has ownership)
		TensorData(const TensorData& other);
		TensorData& operator=(const TensorData& other);
		// Move
		TensorData(TensorData&& other) noexcept;
		TensorData& operator=(TensorData&& other) noexcept;
		// Destructor
		~TensorData() { cleanup(); }

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
		void saveToPath(const std::string& path);
		void loadFromPath(const std::string& path, bool loadWeightsAndBias =false);

		float* operator[](size_t col) { return colData(col); }
		const float* operator[](size_t col) const { return colData(col); }

	protected:
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


