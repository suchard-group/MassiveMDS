#ifndef _OPENCL_INSTANTIATE_CPP
#define _OPENCL_INSTANTIATE_CPP

#include "OpenCLMultiDimensionalScaling.hpp"

namespace mds {
//
// template class OpenCLMultiDimensionalScaling<float>;
// template class OpenCLMultiDimensionalScaling<double>;
//

// factory
std::shared_ptr<AbstractMultiDimensionalScaling>
constructOpenCLMultiDimensionalScalingDouble(int embeddingDimension, int locationCount, long flags) {
	return std::make_shared<OpenCLMultiDimensionalScaling<double>>(embeddingDimension, locationCount, flags);
}

std::shared_ptr<AbstractMultiDimensionalScaling>
constructOpenCLMultiDimensionalScalingFloat(int embeddingDimension, int locationCount, long flags) {
	return std::make_shared<OpenCLMultiDimensionalScaling<float>>(embeddingDimension, locationCount, flags);
}

} // namespace mds

#endif // _OPENCL_INSTANTIATE_CPP
