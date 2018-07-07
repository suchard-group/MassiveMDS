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
    constructOpenCLMultiDimensionalScalingDouble(int embeddingDimension, int locationCount, long flags, int device) {
        if (embeddingDimension <= 2) {
            return std::make_shared<OpenCLMultiDimensionalScaling<OpenCLDouble<2>>>(embeddingDimension, locationCount,
                                                                                    flags, device);
        } else if (embeddingDimension <= 4) {
            return std::make_shared<OpenCLMultiDimensionalScaling<OpenCLDouble<4>>>(embeddingDimension, locationCount,
                                                                                    flags, device);
        } else if (embeddingDimension <= 8) {
            return std::make_shared<OpenCLMultiDimensionalScaling<OpenCLDouble<8>>>(embeddingDimension, locationCount,
                                                                                    flags, device);
        } else {
            exit(-1);
        }
    }

    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructOpenCLMultiDimensionalScalingFloat(int embeddingDimension, int locationCount, long flags, int device) {
        if (embeddingDimension <= 2) {
            return std::make_shared<OpenCLMultiDimensionalScaling<OpenCLFloat<2>>>(embeddingDimension, locationCount,
                                                                                   flags, device);
        } else if (embeddingDimension <= 4) {
            return std::make_shared<OpenCLMultiDimensionalScaling<OpenCLFloat<4>>>(embeddingDimension, locationCount,
                                                                                   flags, device);
        } else if (embeddingDimension <= 8) {
            return std::make_shared<OpenCLMultiDimensionalScaling<OpenCLFloat<8>>>(embeddingDimension, locationCount,
                                                                                   flags, device);
        } else {
            exit(-1);
        }
    }

} // namespace mds

#endif // _OPENCL_INSTANTIATE_CPP
