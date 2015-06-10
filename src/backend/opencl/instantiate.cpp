#ifndef _OPENCL_INSTANTIATE_CPP
#define _OPENCL_INSTANTIATE_CPP

#include "OpenCLMultiDimensionalScaling.hpp"

namespace mds {

template class OpenCLMultiDimensionalScaling<float>;
template class OpenCLMultiDimensionalScaling<double>;

} // namespace mds

#endif // _OPENCL_INSTANTIATE_CPP
