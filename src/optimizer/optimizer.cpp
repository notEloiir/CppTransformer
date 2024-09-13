#include <optimizer/optimizer.h>
#include <compiler_flags.h>


void tfm::Optimizer::clear_gradient(tfm::Tensor& gradient) {
#ifdef SAVE_VRAM
	gradient.move_to(tfm::Device(tfm::DeviceType::CPU));
#endif // SAVE_VRAM
	gradient.fill(0.0f);
}
