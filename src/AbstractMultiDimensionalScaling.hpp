#ifndef _ABSTRACTMULTIDIMENSIONALSCALING_HPP
#define _ABSTRACTMULTIDIMENSIONALSCALING_HPP

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <complex>
#include <future>

//#include <xmmintrin.h>
#include <emmintrin.h>

// #define USE_TBB

#ifdef USE_TBB
    #include "tbb/parallel_reduce.h"
    #include "tbb/blocked_range.h"
#endif

#include "MemoryManagement.hpp"
#include "ThreadPool.h"
#include "CDF.h"
#include "flags.h"

namespace mds {

class AbstractMultiDimensionalScaling {
public:
    AbstractMultiDimensionalScaling(int embeddingDimension, int locationCount, long flags)
        : embeddingDimension(embeddingDimension), locationCount(locationCount),
          observationCount(locationCount * (locationCount - 1) / 2),
          flags(flags) { }

    virtual ~AbstractMultiDimensionalScaling() { }

    // Interface
    virtual void updateLocations(int, double*, size_t) = 0;
    virtual double getSumOfSquaredResiduals() = 0;
    virtual double getSumOfLogTruncations() = 0;
    virtual void storeState() = 0;
    virtual void restoreState() = 0;
    virtual void acceptState() = 0;
    virtual void setPairwiseData(double*, size_t)  = 0;
    virtual void setParameters(double*, size_t) = 0;
    virtual void makeDirty() = 0;

    virtual double getDiagnostic() { return 0.0; }

protected:
    int embeddingDimension;
    int locationCount;
    int observationCount;
    long flags;

    int updatedLocation = -1;
    bool residualsAndTruncationsKnown = false;
    bool sumsOfResidualsAndTruncationsKnown = false;
    bool isLeftTruncated = false;
};

template <typename T>
struct DetermineType;

} // namespace mds

#endif // _ABSTRACTMULTIDIMENSIONALSCALING_HPP
