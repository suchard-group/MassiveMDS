#include "AbstractMultiDimensionalScaling.hpp"


#define USE_SIMD

// forward reference
namespace mds {

    SharedPtr constructOpenCLMultiDimensionalScalingDouble(int, int, long, int);
    SharedPtr constructOpenCLMultiDimensionalScalingFloat(int, int, long, int);

    SharedPtr constructNewMultiDimensionalScalingDoubleNoParallelNoSimd(int, int, long, int);
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbNoSimd(int, int, long, int);

    SharedPtr constructNewMultiDimensionalScalingFloatNoParallelNoSimd(int, int, long, int);
    SharedPtr constructNewMultiDimensionalScalingFloatTbbNoSimd(int, int, long, int);

#ifdef USE_SIMD
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbSimd(int, int, long, int);
    SharedPtr constructNewMultiDimensionalScalingDoubleNoParallelSimd(int, int, long, int);
    SharedPtr constructNewMultiDimensionalScalingFloatNoParallelSimd(int, int, long, int);
    SharedPtr constructNewMultiDimensionalScalingFloatTbbSimd(int, int, long, int);
#endif

SharedPtr factory(int dim1, int dim2, long flags, int device, int threads) {
	bool useFloat = flags & mds::Flags::FLOAT;
	bool useOpenCL = flags & mds::Flags::OPENCL;
	bool useTbb = flags & mds::Flags::TBB;
	bool useSimd = flags & mds::Flags::SIMD;

	if (useFloat) {
		if (useOpenCL) {
			return constructOpenCLMultiDimensionalScalingFloat(dim1, dim2, flags, device);
		} else {
#ifdef USE_SIMD
		    if (useSimd) {
                if (useTbb) {
                    return constructNewMultiDimensionalScalingFloatTbbSimd(dim1, dim2, flags, threads);
                } else {
                    return constructNewMultiDimensionalScalingFloatNoParallelSimd(dim1, dim2, flags, threads);
                }
		    } else {
#endif
                if (useTbb) {
                    return constructNewMultiDimensionalScalingFloatTbbNoSimd(dim1, dim2, flags, threads);
                } else {
                    return constructNewMultiDimensionalScalingFloatNoParallelNoSimd(dim1, dim2, flags, threads);
                }
#ifdef USE_SIMD
            }
#endif
		}
	} else {
		if (useOpenCL) {
			return constructOpenCLMultiDimensionalScalingDouble(dim1, dim2, flags, device);
		} else {
#ifdef USE_SIMD
		    if (useSimd) {
                if (useTbb) {
                    return constructNewMultiDimensionalScalingDoubleTbbSimd(dim1, dim2, flags, threads);
                } else {
                    return constructNewMultiDimensionalScalingDoubleNoParallelSimd(dim1, dim2, flags, threads);
                }
		    } else {
#endif
                if (useTbb) {
                    return constructNewMultiDimensionalScalingDoubleTbbNoSimd(dim1, dim2, flags, threads);
                } else {
                    return constructNewMultiDimensionalScalingDoubleNoParallelNoSimd(dim1, dim2, flags, threads);
                }
#ifdef USE_SIMD
            }
#endif
		}
	}
}

} // namespace mds