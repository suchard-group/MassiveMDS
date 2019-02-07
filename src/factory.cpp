#include "AbstractMultiDimensionalScaling.hpp"

// forward reference
namespace mds {

    SharedPtr constructOpenCLMultiDimensionalScalingDouble(int, int, long, int);
    SharedPtr constructOpenCLMultiDimensionalScalingFloat(int, int, long, int);

    SharedPtr constructNewMultiDimensionalScalingDoubleNoParallelNoSimd(int, int, long, int);
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbNoSimd(int, int, long, int);

    SharedPtr constructNewMultiDimensionalScalingFloatNoParallelNoSimd(int, int, long, int);
    SharedPtr constructNewMultiDimensionalScalingFloatTbbNoSimd(int, int, long, int);

#ifdef USE_SIMD
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbAvx(int, int, long, int);
    SharedPtr constructNewMultiDimensionalScalingDoubleNoParallelAvx(int, int, long, int);
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbSse(int, int, long, int);
    SharedPtr constructNewMultiDimensionalScalingDoubleNoParallelSse(int, int, long, int);
    SharedPtr constructNewMultiDimensionalScalingFloatNoParallelSse(int, int, long, int);
    SharedPtr constructNewMultiDimensionalScalingFloatTbbSse(int, int, long, int);
#endif

SharedPtr factory(int dim1, int dim2, long flags, int device, int threads) {
	bool useFloat = flags & mds::Flags::FLOAT;
	bool useOpenCL = flags & mds::Flags::OPENCL;
	bool useTbb = flags & mds::Flags::TBB;
	bool useAvx = flags & mds::Flags::AVX;
	bool useSse = flags & mds::Flags::SSE;

	if (useFloat) {
		if (useOpenCL) {
			return constructOpenCLMultiDimensionalScalingFloat(dim1, dim2, flags, device);
		} else {
#ifdef USE_SIMD
		    if (useSse) {
                if (useTbb) {
                    return constructNewMultiDimensionalScalingFloatTbbSse(dim1, dim2, flags, threads);
                } else {
                    return constructNewMultiDimensionalScalingFloatNoParallelSse(dim1, dim2, flags, threads);
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
		    if (useAvx) {
                if (useTbb) {
                    return constructNewMultiDimensionalScalingDoubleTbbAvx(dim1, dim2, flags, threads);
                } else {
                    return constructNewMultiDimensionalScalingDoubleNoParallelAvx(dim1, dim2, flags, threads);
                }
		    } else if (useSse) {
                if (useTbb) {
                    return constructNewMultiDimensionalScalingDoubleTbbSse(dim1, dim2, flags, threads);
                } else {
                    return constructNewMultiDimensionalScalingDoubleNoParallelSse(dim1, dim2, flags, threads);
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