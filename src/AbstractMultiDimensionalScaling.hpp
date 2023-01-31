#ifndef _ABSTRACT_MULTIDIMENSIONAL_SCALING_HPP
#define _ABSTRACT_MULTIDIMENSIONAL_SCALING_HPP

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <complex>
#include <future>


#ifdef USE_SIMD
  #if defined(__ARM64_ARCH_8__)
    #include "sse2neon.h"
    #undef USE_AVX
    #undef USE_AVX512
  #else
    #include <emmintrin.h>
    #include <smmintrin.h>
  #endif
#endif

#define USE_TBB

#ifdef USE_TBB
    #include "tbb/parallel_reduce.h"
    #include "tbb/blocked_range.h"
    #include "tbb/parallel_for.h"
#endif

#include "MemoryManagement.hpp"
#include "ThreadPool.h"
#include "CDF.h"
#include "flags.h"

namespace mds {

class AbstractMultiDimensionalScaling {
public:
    AbstractMultiDimensionalScaling(int embeddingDimension, int locationCount, long flags, int bandwidth)
        : embeddingDimension(embeddingDimension), locationCount(locationCount),
          observationCount(locationCount * (locationCount - 1) / 2),
          flags(flags),
          bandwidth(bandwidth) { }

    virtual ~AbstractMultiDimensionalScaling() = default;

    // Interface
    virtual void updateLocations(int, double*, size_t) = 0;
//     virtual double getSumOfSquaredResiduals() = 0;
//     virtual double getSumOfLogTruncations() = 0;
    virtual double getSumOfIncrements() = 0;
    virtual void getLogLikelihoodGradient(double*, size_t) = 0;
    virtual void storeState() = 0;
    virtual void restoreState() = 0;
    virtual void acceptState() = 0;
    virtual void setPairwiseData(double*, size_t)  = 0;
    virtual void setParameters(double*, size_t) = 0;
    virtual void makeDirty() = 0;
    virtual int getInternalDimension() = 0;

//    virtual double getDiagnostic() { return 0.0; }

protected:
    int embeddingDimension;
    int locationCount;
    int bandwidth;
    int observationCount;
    long flags;

    int updatedLocation = -1;
    bool incrementsKnown = false;
    bool sumOfIncrementsKnown = false;
    bool isLeftTruncated = false;
};

typedef std::shared_ptr<mds::AbstractMultiDimensionalScaling> SharedPtr;

SharedPtr factory(int dim1, int dim2, long flags, int device, int threads, int bandwidth);

//template <typename T>
//struct DetermineType;

struct CpuAccumulate { };

#ifdef USE_TBB
struct TbbAccumulate{ };
#endif


} // namespace mds

#endif // _ABSTRACT_MULTIDIMENSIONAL_SCALING_HPP
