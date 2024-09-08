#pragma once
#include <compiler_flags.h>


namespace tfm {

enum DeviceType {
	CPU,
	CUDA
};

class Device {
public:
	Device(DeviceType type, int index=0) noexcept;

	bool is_CPU() const { return type_ == tfm::DeviceType::CPU; }
	bool is_CUDA() const { return type_ == tfm::DeviceType::CUDA; }
	int index() const { return id_; }
	bool operator==(const Device& other) const { return type_ == other.type_ && id_ == other.id_; }

	static const int device_count;

private:
	tfm::DeviceType type_;
	int id_;

};

}



