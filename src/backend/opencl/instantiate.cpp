#ifdef HAVE_OPENCL
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
    constructOpenCLMultiDimensionalScalingDouble(int embeddingDimension, Layout layout, long flags, int device) {
        if (embeddingDimension <= 2) {
            return std::make_shared<OpenCLMultiDimensionalScaling<OpenCLDouble<2>>>(embeddingDimension, layout,
                                                                                    flags, device);
        } else if (embeddingDimension <= 4) {
            return std::make_shared<OpenCLMultiDimensionalScaling<OpenCLDouble<4>>>(embeddingDimension, layout,
                                                                                    flags, device);
        } else if (embeddingDimension <= 8) {
            return std::make_shared<OpenCLMultiDimensionalScaling<OpenCLDouble<8>>>(embeddingDimension, layout,
                                                                                    flags, device);
        } else {
#ifdef RBUILD
            Rcpp::stop("Embedding dimension > 8!\n");
#else
            exit(-1);
#endif
        }
    }

    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructOpenCLMultiDimensionalScalingFloat(int embeddingDimension, Layout layout, long flags, int device) {
        if (embeddingDimension <= 2) {
            return std::make_shared<OpenCLMultiDimensionalScaling<OpenCLFloat<2>>>(embeddingDimension, layout,
                                                                                   flags, device);
        } else if (embeddingDimension <= 4) {
            return std::make_shared<OpenCLMultiDimensionalScaling<OpenCLFloat<4>>>(embeddingDimension, layout,
                                                                                   flags, device);
        } else if (embeddingDimension <= 8) {
            return std::make_shared<OpenCLMultiDimensionalScaling<OpenCLFloat<8>>>(embeddingDimension, layout,
                                                                                   flags, device);
        } else {
#ifdef RBUILD
            Rcpp::stop("Embedding dimension > 8!\n");
#else
            exit(-1);
#endif
        }
    }

} // namespace mds

#endif // _OPENCL_INSTANTIATE_CPP
#endif
