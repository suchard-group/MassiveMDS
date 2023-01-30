#include "AbstractMultiDimensionalScaling.hpp"

// forward reference
namespace mds {
#ifdef HAVE_OPENCL
    SharedPtr constructOpenCLMultiDimensionalScalingDouble(int, int, long, int, int);
    SharedPtr constructOpenCLMultiDimensionalScalingFloat(int, int, long, int, int);
#endif
    SharedPtr constructNewMultiDimensionalScalingDoubleNoParallelNoSimd(int, int, long, int, int);
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbNoSimd(int, int, long, int, int);

    SharedPtr constructNewMultiDimensionalScalingFloatNoParallelNoSimd(int, int, long, int, int);
    SharedPtr constructNewMultiDimensionalScalingFloatTbbNoSimd(int, int, long, int, int);

#ifdef USE_SSE
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbSse(int, int, long, int, int);
    SharedPtr constructNewMultiDimensionalScalingDoubleNoParallelSse(int, int, long, int, int);
    SharedPtr constructNewMultiDimensionalScalingFloatNoParallelSse(int, int, long, int, int);
    SharedPtr constructNewMultiDimensionalScalingFloatTbbSse(int, int, long, int, int);
#endif

#ifdef USE_AVX
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbAvx(int, int, long, int, int);
    SharedPtr constructNewMultiDimensionalScalingDoubleNoParallelAvx(int, int, long, int, int);
#endif

#ifdef USE_AVX512
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbAvx512(int, int, long, int, int);
    SharedPtr constructNewMultiDimensionalScalingDoubleNoParallelAvx512(int, int, long, int, int);
#endif

SharedPtr factory(int dim1, int dim2, long flags, int device, int threads, int bandwidth) {
	bool useFloat = flags & mds::Flags::FLOAT;
	bool useOpenCL = flags & mds::Flags::OPENCL;
	bool useTbb = flags & mds::Flags::TBB;
    bool useAvx512 = flags & mds::Flags::AVX512;
	bool useAvx = flags & mds::Flags::AVX;
	bool useSse = flags & mds::Flags::SSE;

	if (useFloat) {
		if (useOpenCL) {
#ifdef HAVE_OPENCL
			return constructOpenCLMultiDimensionalScalingFloat(dim1, dim2, flags, device, bandwidth);
#else
		  return constructNewMultiDimensionalScalingFloatNoParallelNoSimd(dim1, dim2, flags, threads, bandwidth);
#endif
		} else {
#ifdef USE_SSE
		    if (useSse) {
                if (useTbb) {
                    return constructNewMultiDimensionalScalingFloatTbbSse(dim1, dim2, flags, threads, bandwidth);
                } else {
                    return constructNewMultiDimensionalScalingFloatNoParallelSse(dim1, dim2, flags, threads, bandwidth);
                }
		    } else {
#endif
                if (useTbb) {
                    return constructNewMultiDimensionalScalingFloatTbbNoSimd(dim1, dim2, flags, threads, bandwidth);
                } else {
                    return constructNewMultiDimensionalScalingFloatNoParallelNoSimd(dim1, dim2, flags, threads, bandwidth);
                }
#ifdef USE_SSE
            }
#endif
		}
	} else {
		if (useOpenCL) {
#ifdef HAVE_OPENCL
			return constructOpenCLMultiDimensionalScalingDouble(dim1, dim2, flags, device, bandwidth);
#else
		  return constructNewMultiDimensionalScalingDoubleNoParallelNoSimd(dim1, dim2, flags, threads, bandwidth);
#endif
		} else {

#ifdef USE_AVX512
            if (useAvx512) {
                if (useTbb) {
                    return constructNewMultiDimensionalScalingDoubleTbbAvx512(dim1, dim2, flags, threads, bandwidth);
                } else {
                    return constructNewMultiDimensionalScalingDoubleNoParallelAvx512(dim1, dim2, flags, threads, bandwidth);
                }
            } else
#else
              useAvx512 = false; // stops unused variable warning when AVX512 is unavailable
#endif // USE_AVX512

#ifdef USE_AVX
		    if (useAvx) {
                if (useTbb) {
                    return constructNewMultiDimensionalScalingDoubleTbbAvx(dim1, dim2, flags, threads, bandwidth);
                } else {
                    return constructNewMultiDimensionalScalingDoubleNoParallelAvx(dim1, dim2, flags, threads, bandwidth);
                }
            } else
#else
              useAvx = false;
#endif // USE_AVX

#ifdef USE_SSE
		    if (useSse) {
                if (useTbb) {
                    return constructNewMultiDimensionalScalingDoubleTbbSse(dim1, dim2, flags, threads, bandwidth);
                } else {
                    return constructNewMultiDimensionalScalingDoubleNoParallelSse(dim1, dim2, flags, threads, bandwidth);
                }
		    } else
#endif // USE_SSE
            {

                if (useTbb) {
                    return constructNewMultiDimensionalScalingDoubleTbbNoSimd(dim1, dim2, flags, threads, bandwidth);
                } else {
                    return constructNewMultiDimensionalScalingDoubleNoParallelNoSimd(dim1, dim2, flags, threads, bandwidth);
                }
            }
		}
	}
}

} // namespace mds
