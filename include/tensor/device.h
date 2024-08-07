#pragma once



namespace tfm
{

enum DeviceType {
	CPU,
	CUDA
};

class Device {
public:
	Device(DeviceType type, int index=0) noexcept;

	bool isCPU() const { return type == tfm::DeviceType::CPU; }
	bool isCUDA() const { return type == tfm::DeviceType::CUDA; }
	int index() const { return id; }
	bool operator==(const Device& other) const { return type == other.type && id == other.id; }

	static const int deviceCount;

private:
	tfm::DeviceType type;
	int id;

};

}



