#include "AbstractMultiDimensionalScaling.hpp"

// forward reference
namespace mds {
#ifdef HAVE_OPENCL
    SharedPtr constructOpenCLMultiDimensionalScalingDouble(int, const Layout&, long, int);
    SharedPtr constructOpenCLMultiDimensionalScalingFloat(int, const Layout&, long, int);
#endif
    SharedPtr constructNewMultiDimensionalScalingDoubleNoParallelNoSimd(int, const Layout&, long, int);
    SharedPtr constructNewMultiDimensionalScalingFloatNoParallelNoSimd(int, const Layout&, long, int);
#ifdef USE_TBB
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbNoSimd(int, const Layout&, long, int);
    SharedPtr constructNewMultiDimensionalScalingFloatTbbNoSimd(int, const Layout&, long, int);
#endif

#ifdef USE_SSE
    SharedPtr constructNewMultiDimensionalScalingDoubleNoParallelSse(int, const Layout&, long, int);
    SharedPtr constructNewMultiDimensionalScalingFloatNoParallelSse(int, const Layout&, long, int);
#ifdef USE_TBB
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbSse(int, const Layout&, long, int);
    SharedPtr constructNewMultiDimensionalScalingFloatTbbSse(int, const Layout&, long, int);
#endif
#endif

#ifdef USE_AVX
#ifdef USE_TBB
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbAvx(int, const Layout&, long, int);
#endif
    SharedPtr constructNewMultiDimensionalScalingDoubleNoParallelAvx(int, const Layout&, long, int);
#endif

#ifdef USE_AVX512
#ifdef USE_TBB
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbAvx512(int, const Layout&, long, int);
#endif
    SharedPtr constructNewMultiDimensionalScalingDoubleNoParallelAvx512(int, const Layout&, long, int);
#endif

#ifndef USE_TBB
	SharedPtr constructNewMultiDimensionalScalingDoubleTbbNoSimd(int, const Layout&, long, int) { return nullptr; }
    SharedPtr constructNewMultiDimensionalScalingFloatTbbNoSimd(int, const Layout&, long, int) { return nullptr; }
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbSse(int, const Layout&, long, int) { return nullptr; }
    SharedPtr constructNewMultiDimensionalScalingFloatTbbSse(int, const Layout&, long, int) { return nullptr; }
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbAvx(int, const Layout&, long, int) { return nullptr; }
    SharedPtr constructNewMultiDimensionalScalingDoubleTbbAvx512(int, const Layout&, long, int) { return nullptr; }
#endif

SharedPtr factory(int dim1, const Layout& layout, long flags, int device, int threads) {
	bool useFloat = flags & mds::Flags::FLOAT;
	bool useOpenCL = flags & mds::Flags::OPENCL;
	bool useTbb = flags & mds::Flags::TBB;
    __attribute__((unused))
    bool useAvx512 = flags & mds::Flags::AVX512;
    __attribute__((unused))
    bool useAvx = flags & mds::Flags::AVX;
	bool useSse = flags & mds::Flags::SSE;

	if (useFloat) {
		if (useOpenCL) {
#ifdef HAVE_OPENCL
			return constructOpenCLMultiDimensionalScalingFloat(dim1, layout, flags, device);
#else
            std::cerr << "Not compiled with OpenCL" << std::endl;
            exit(-1);
		  return constructNewMultiDimensionalScalingFloatNoParallelNoSimd(dim1, layout, flags, threads);
#endif
		} else {
#ifdef USE_SSE
		    if (useSse) {
                if (useTbb) {
                    return constructNewMultiDimensionalScalingFloatTbbSse(dim1, layout, flags, threads);
                } else {
                    return constructNewMultiDimensionalScalingFloatNoParallelSse(dim1, layout, flags, threads);
                }
		    } else {
#endif
                if (useTbb) {
                    return constructNewMultiDimensionalScalingFloatTbbNoSimd(dim1, layout, flags, threads);
                } else {
                    return constructNewMultiDimensionalScalingFloatNoParallelNoSimd(dim1, layout, flags, threads);
                }
#ifdef USE_SSE
            }
#endif
		}
	} else {
		if (useOpenCL) {
#ifdef HAVE_OPENCL
			return constructOpenCLMultiDimensionalScalingDouble(dim1, layout, flags, device);
#else
            std::cerr << "Not compiled with OpenCL" << std::endl;
            exit(-1);
		  return constructNewMultiDimensionalScalingDoubleNoParallelNoSimd(dim1, layout, flags, threads);
#endif
		} else {

#ifdef USE_AVX512
            if (useAvx512) {
                if (useTbb) {
                    return constructNewMultiDimensionalScalingDoubleTbbAvx512(dim1, layout, flags, threads);
                } else {
                    return constructNewMultiDimensionalScalingDoubleNoParallelAvx512(dim1, layout, flags, threads);
                }
            } else
#else
//              useAvx512 = false; // stops unused variable warning when AVX512 is unavailable
#endif // USE_AVX512

#ifdef USE_AVX
		    if (useAvx) {
                if (useTbb) {
                    return constructNewMultiDimensionalScalingDoubleTbbAvx(dim1, layout, flags, threads);
                } else {
                    return constructNewMultiDimensionalScalingDoubleNoParallelAvx(dim1, layout, flags, threads);
                }
            } else
#else
//              useAvx = false;
#endif // USE_AVX

#ifdef USE_SSE
		    if (useSse) {
                if (useTbb) {
                    return constructNewMultiDimensionalScalingDoubleTbbSse(dim1, layout, flags, threads);
                } else {
                    return constructNewMultiDimensionalScalingDoubleNoParallelSse(dim1, layout, flags, threads);
                }
		    } else
#endif // USE_SSE
            {

                if (useTbb) {
                    return constructNewMultiDimensionalScalingDoubleTbbNoSimd(dim1, layout, flags, threads);
                } else {
                    return constructNewMultiDimensionalScalingDoubleNoParallelNoSimd(dim1, layout, flags, threads);
                }
            }
		}
	}
}

} // namespace mds
