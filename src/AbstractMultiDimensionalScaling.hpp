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

template <typename RealType>
class MultiDimensionalScaling : public AbstractMultiDimensionalScaling {
public:
    MultiDimensionalScaling(int embeddingDimension, int locationCount, long flags)
        : AbstractMultiDimensionalScaling(embeddingDimension, locationCount, flags),
          precision(0.0), storedPrecision(0.0),
          oneOverSd(0.0), storedOneOverSd(0.0),
          sumOfSquaredResiduals(0.0), storedSumOfSquaredResiduals(0.0),
          sumOfTruncations(0.0), storedSumOfTruncations(0.0),

          observations(locationCount * locationCount),

          locations0(locationCount * embeddingDimension),
		  locations1(locationCount * embeddingDimension),
		  locationsPtr(&locations0),
		  storedLocationsPtr(&locations1),

          squaredResiduals(locationCount * locationCount),
          storedSquaredResiduals(locationCount),

          isStoredSquaredResidualsEmpty(false),
          isStoredTruncationsEmpty(false),
          isStoredAllTruncationsEmpty(false)

          , nThreads(4) //, pool(nThreads)
    {

    	if (flags & Flags::LEFT_TRUNCATION) {
    		isLeftTruncated = true;
    		std::cout << "Using left truncation" << std::endl;

    		truncations.resize(locationCount * locationCount);
    		storedTruncations.resize(locationCount);
    	}
    }

    virtual ~MultiDimensionalScaling() { }

    void updateLocations(int locationIndex, double* location, size_t length) {

    	assert(length == embeddingDimension);

    	if (updatedLocation != - 1) {
    		// more than one location updated -- do a full recomputation
    		residualsAndTruncationsKnown = false;
    		isStoredSquaredResidualsEmpty = true;
    		isStoredTruncationsEmpty = true;
    	}

    	updatedLocation = locationIndex;
    	std::copy(location, location + length,
    		begin(*locationsPtr) + locationIndex * embeddingDimension
    	);

    	sumsOfResidualsAndTruncationsKnown = false;
    }

    void computeResidualsAndTruncations() {

		if (!residualsAndTruncationsKnown) {
			if (isLeftTruncated) { // run-time dispatch to compile-time optimization
				computeSumOfSquaredResiduals<true>();
			} else {
				computeSumOfSquaredResiduals<false>();
			}
			residualsAndTruncationsKnown = true;
		} else {
			if (isLeftTruncated) {
#if 1
				updateSumOfSquaredResidualsAndTruncations();
#else
				updateSumOfSquaredResiduals();
				updateTruncations();
#endif
			} else {
				updateSumOfSquaredResiduals();
			}
		}
    }

    double getSumOfSquaredResiduals() {
    	if (!sumsOfResidualsAndTruncationsKnown) {
			computeResidualsAndTruncations();
			sumsOfResidualsAndTruncationsKnown = true;
		}
		return sumOfSquaredResiduals;
 	}

 	double getSumOfLogTruncations() {
    	if (!sumsOfResidualsAndTruncationsKnown) {
			computeResidualsAndTruncations();
			sumsOfResidualsAndTruncationsKnown = true;
		}
 		return sumOfTruncations;
 	}

    void storeState() {
    	storedSumOfSquaredResiduals = sumOfSquaredResiduals;

    	std::copy(begin(*locationsPtr), end(*locationsPtr),
    		begin(*storedLocationsPtr));

    	isStoredSquaredResidualsEmpty = true;

    	updatedLocation = -1;

    	storedPrecision = precision;
    	storedOneOverSd = oneOverSd;

    	// Handle truncation
    	if (isLeftTruncated) {
    		storedSumOfTruncations = sumOfTruncations;
			isStoredTruncationsEmpty = true;
    	}
    }

    void acceptState() {
        // Do nothing
    }

    void restoreState() {
    	sumOfSquaredResiduals = storedSumOfSquaredResiduals;
    	sumsOfResidualsAndTruncationsKnown = true;

		if (!isStoredSquaredResidualsEmpty) {
    		std::copy(
    			begin(storedSquaredResiduals),
    			end(storedSquaredResiduals),
    			begin(squaredResiduals) + updatedLocation * locationCount
    		);
    		for (int j = 0; j < locationCount; ++j) {
    			squaredResiduals[j * locationCount + updatedLocation]
    				= storedSquaredResiduals[j];
    		}
    		residualsAndTruncationsKnown = true;
    	} else {
    		residualsAndTruncationsKnown = false; // Force recompute;  TODO cache
    	}

    	// Handle truncation
    	if (isLeftTruncated) {
	    	sumOfTruncations = storedSumOfTruncations;

	    	if (!isStoredTruncationsEmpty) {
	    		std::copy(
	    			begin(storedTruncations),
	    			end(storedTruncations),
	    			begin(truncations) + updatedLocation * locationCount
	    		);
	    		for (int j = 0; j < locationCount; ++j) {
	    			truncations[j * locationCount + updatedLocation]
	    				= storedTruncations[j];
	    		}
	    	}
	    }

    	precision = storedPrecision;
    	oneOverSd = storedOneOverSd;

    	auto tmp1 = storedLocationsPtr;
    	storedLocationsPtr = locationsPtr;
    	locationsPtr = tmp1;
    }

    void setPairwiseData(double* data, size_t length) {
		assert(length == observations.size());
		std::copy(data, data + length, begin(observations));
    }

    void setParameters(double* data, size_t length) {
		assert(length == 1); // Call only with precision
		precision = data[0]; // TODO Remove
		oneOverSd = std::sqrt(data[0]);

		// Handle truncations
		if (isLeftTruncated) {
			residualsAndTruncationsKnown = false;
			sumsOfResidualsAndTruncationsKnown = false;

    		isStoredSquaredResidualsEmpty = true;
    		isStoredTruncationsEmpty = true;

		}
    }

    void makeDirty() {
    	sumsOfResidualsAndTruncationsKnown = false;
    	residualsAndTruncationsKnown = false;
    }


	int count = 0;

	template <bool withTruncation>
	void computeSumOfSquaredResiduals() {

		RealType lSumOfSquaredResiduals = 0.0;
		RealType lSumOfTruncations = 0.0;

		for (int i = 0; i < locationCount; ++i) { // TODO Parallelize
			for (int j = 0; j < locationCount; ++j) { // TODO j = i + 1 ???

				const auto distance = calculateDistance<mm::MemoryManager<RealType>>(
					begin(*locationsPtr) + i * embeddingDimension,
					begin(*locationsPtr) + j * embeddingDimension,
					embeddingDimension
				);
				const auto residual = distance - observations[i * locationCount + j];
				const auto squaredResidual = residual * residual;
				squaredResiduals[i * locationCount + j] = squaredResidual;
// 				squaredResiduals[j * locationCount + i] = squaredResidual;
				lSumOfSquaredResiduals += squaredResidual;

				if (withTruncation) { // compile-time check
					const auto truncation = (i == j) ? RealType(0) :
						math::logCdf(std::fabs(residual) * oneOverSd);
					truncations[i * locationCount + j] = truncation;
// 					truncations[j * locationCount + i] = truncation;
					lSumOfTruncations += truncation;
				}
			}
		}

    	lSumOfSquaredResiduals /= 2.0;
    	sumOfSquaredResiduals = lSumOfSquaredResiduals;

    	if (withTruncation) {
    		lSumOfTruncations /= 2.0;
    		sumOfTruncations = lSumOfTruncations;
    	}

	    residualsAndTruncationsKnown = true;
	    sumsOfResidualsAndTruncationsKnown = true;
	}

	void updateSumOfSquaredResiduals() {
		// double delta = 0.0;

		const int i = updatedLocation;
		isStoredSquaredResidualsEmpty = false;

		auto start  = begin(*locationsPtr) + i * embeddingDimension;
		auto offset = begin(*locationsPtr);

		RealType delta =

// 		accumulate_thread(0, locationCount, double(0),
 		accumulate(0, locationCount, RealType(0),
// 		accumulate_tbb(0, locationCount, double(0),

			[this, i, &offset,
			&start](const int j) {
                const auto distance = calculateDistance<mm::MemoryManager<RealType>>(
                    start,
                    offset + embeddingDimension * j,
                    embeddingDimension
                );

                const auto residual = distance - observations[i * locationCount + j];
                const auto squaredResidual = residual * residual;

            	// store old value
            	const auto oldSquaredResidual = squaredResiduals[i * locationCount + j];
            	storedSquaredResiduals[j] = oldSquaredResidual;

                const auto inc = squaredResidual - oldSquaredResidual;

            	// store new value
                squaredResiduals[i * locationCount + j] = squaredResidual;
                squaredResiduals[j * locationCount + i] = squaredResidual;

                return inc;
            }
		);


// #if 1
// #if 1
//         for_each(0, locationCount,
// #else
//         for_each_auto<2>(0, locationCount,
// #endif
//             [this, i, &delta, &offset, &start](const int j) {
//                 const auto distance = calculateDistance(
//                     start,
//                     offset,
//                     embeddingDimension
//                 );
//                 offset += embeddingDimension;
//
//                 const auto residual = distance - observations[i * locationCount + j];
//                 const auto squaredResidual = residual * residual;
//
//                 delta += squaredResidual - squaredResiduals[i * locationCount + j];
//
//                 squaredResiduals[i * locationCount + j] = squaredResidual;
//                 squaredResiduals[j * locationCount + i] = squaredResidual;
//             }
//         );
// #else
// 		for (int j = 0; j < locationCount; ++j) {
// 			// TODO Code duplication with above
// 			const auto distance = calculateDistance(
// 				begin(*locationsPtr) + i * embeddingDimension,
// 				begin(*locationsPtr) + j * embeddingDimension,
// 				embeddingDimension
// 			);
// 			const auto residual = distance - observations[i * locationCount + j];
// 			const auto squaredResidual = residual * residual;
//
// 			delta += squaredResidual - squaredResiduals[i * locationCount + j];
//
// 			squaredResiduals[i * locationCount + j] = squaredResidual;
// 			squaredResiduals[j * locationCount + i] = squaredResidual;
// 		}
// #endif

		sumOfSquaredResiduals += delta;
	}

// 	int count = 0
	int count2 = 0;


	void updateTruncations() {

		const int i = updatedLocation;

		isStoredTruncationsEmpty = false;

		auto start  = begin(*locationsPtr) + i * embeddingDimension;
		auto offset = begin(*locationsPtr);

		RealType delta =
// 		accumulate_thread(0, locationCount, double(0),
 		accumulate(0, locationCount, RealType(0),
// 		accumulate_tbb(0, locationCount, double(0),

			[this, i, &offset, //oneOverSd,
			&start](const int j) {

                const auto squaredResidual = squaredResiduals[i * locationCount + j];

                const auto truncation = (i == j) ? RealType(0) :
                	math::logCdf(std::sqrt(squaredResidual) * oneOverSd);

                const auto oldTruncation = truncations[i * locationCount + j];
                storedTruncations[j] = oldTruncation;

                const auto inc = truncation - oldTruncation;

                truncations[i * locationCount + j] = truncation;
                truncations[j * locationCount + i] = truncation;

                return inc;
            }
		);

 		sumOfTruncations += delta;
	}


	void updateSumOfSquaredResidualsAndTruncations() {

		const int i = updatedLocation;
		isStoredSquaredResidualsEmpty = false;
		isStoredTruncationsEmpty = false;

		auto start  = begin(*locationsPtr) + i * embeddingDimension;
		auto offset = begin(*locationsPtr);

		std::complex<RealType> delta =

// 		accumulate_thread(0, locationCount, double(0),

#ifdef USE_TBB
        accumulate_tbb(0, locationCount, std::complex<RealType>(RealType(0), RealType(0)),
#else
 		accumulate(0, locationCount, std::complex<RealType>(RealType(0), RealType(0)),
#endif
// 		accumulate_tbb(0, locationCount, double(0),

			[this, i, &offset, //oneOverSd,
			&start](const int j) {
                const auto distance = calculateDistance<mm::MemoryManager<RealType>>(
                    start,
                    offset + embeddingDimension * j,
                    embeddingDimension
                );

                const auto residual = distance - observations[i * locationCount + j];
                const auto squaredResidual = residual * residual;

            	// store old value
            	const auto oldSquaredResidual = squaredResiduals[i * locationCount + j];
            	storedSquaredResiduals[j] = oldSquaredResidual;

                const auto inc = squaredResidual - oldSquaredResidual;

            	// store new value
                squaredResiduals[i * locationCount + j] = squaredResidual;
                squaredResiduals[j * locationCount + i] = squaredResidual;

                const auto truncation = (i == j) ? RealType(0) :
                	math::logCdf(std::fabs(residual) * oneOverSd);

                const auto oldTruncation = truncations[i * locationCount + j];
                storedTruncations[j] = oldTruncation;

                const auto inc2 = truncation - oldTruncation;

                truncations[i * locationCount + j] = truncation;
                truncations[j * locationCount + i] = truncation;

                return std::complex<RealType>(inc, inc2);
            }
		);

		sumOfSquaredResiduals += delta.real();
 		sumOfTruncations += delta.imag();
	}

#if 0
	template <typename Iterator>
	double calculateDistance(Iterator x, Iterator y, int length) const
		//-> decltype(Iterator::value_type)
		{

		auto sum = //Iterator::value_type(0);
					static_cast<double>(0);
		for (int i = 0; i < length; ++i, ++x, ++y) {
			const auto difference = *x - *y;
			sum += difference * difference;
		}
		return std::sqrt(sum);
	}
#else

//
// //#define VECTOR
// #ifdef VECTOR
//     double calculateDistance(double* x, double* y, int length) const {
// //     std::cerr << "A";
//         using Vec2 = __m128d;
//
//         Vec2 vecX = _mm_load_pd(x);
//         Vec2 vecY = _mm_load_pd(y);
//         vecX = _mm_sub_pd(vecX, vecY);
//         vecX = _mm_mul_pd(vecX, vecX);
// #if 1
//         double r[2];
//         _mm_store_pd(r,vecX);
//
//         return std::sqrt(r[0] + r[1]);
// #else
//         double r;
//         vecX =  _mm_hadd_pd(vecX, vecX);
//         _mm_store_ps(r, vecX);
//         return std::sqrt(r);
// #endif
//     }
// #else
//     double calculateDistance(double* x, double* y, int length) const {
// //          std::cerr << "B";
//         double r = 0.0;
//         for (int i = 0; i < 2; ++i, ++x, ++y) {
//             const auto difference = *x - *y;
//             r += difference * difference;
//         }
//         return std::sqrt(r);
//
//     }
// #endif // VECTOR

//     double calculateDistance(double* iX, double* iY, int length) const {
//         auto sum = static_cast<double>(0);
//
//         typedef double more_aligned_double __attribute__ ((aligned (16)));
//
//         double* x = iX;
//         double* y = iY;
//
//         #pragma clang loop vectorize(enable) interleave(enable)
//         for (int i = 0; i < 2; ++i, ++x, ++y) {
//             const auto difference = *x - *y;
//             sum += difference * difference;
//         }
//         return std::sqrt(sum);
// //         const auto difference1 = *x - *y;
// //         ++x; ++y;
// //         const auto difference2 = *x - *y;
// //         return std::sqrt(difference1 * difference1 + difference2 * difference2);
//     }

#ifdef SSE
    template <typename VectorType, typename Iterator>
    double calculateDistance(Iterator iX, Iterator iY, int length) const {
        auto sum = static_cast<double>(0);

        using AlignedValueType = typename VectorType::allocator_type::aligned_value_type;

        AlignedValueType* x = &*iX;
        AlignedValueType* y = &*iY;

        for (int i = 0; i < 2; ++i, ++x, ++y) {
            const auto difference = *x - *y;
            sum += difference * difference;
        }
        return std::sqrt(sum);
    }
#else // SSE
    template <typename VectorType, typename Iterator>
    double calculateDistance(Iterator x, Iterator y, int length) const {
        auto sum = static_cast<double>(0);

        for (int i = 0; i < 2; ++i, ++x, ++y) {
            const auto difference = *x - *y;
            sum += difference * difference;
        }
        return std::sqrt(sum);
    }
#endif // SSE

#endif

#if 0

    template <size_t N> struct uint_{ };

    template <size_t N, typename Lambda, typename IterT>
    inline void unroller(const Lambda& f, const IterT& iter, uint_<N>) {
        unroller(f, iter, uint_<N-1>());
        f(iter + N);
    }

    template <typename Lambda, typename IterT>
    inline void unroller(const Lambda& f, const IterT& iter, uint_<0>) {
        f(iter);
    }

	template <typename Integer, typename Function>
	inline void for_each(Integer begin, Integer end, Function function) {
	    for (; begin != end; begin += 4) {
	        function(begin + 0);
	        function(begin + 1);
	        function(begin + 2);
	        function(begin + 3);
	    }
	}

	template <size_t UnrollFact, typename Integer, typename Function>
	inline void for_each_auto(Integer begin, Integer end, Function function) {
	    for (; begin != end; begin += UnrollFact) {
	        unroller(function, begin,  uint_<UnrollFact-1>());
	    }
	}
#else
	template <typename Integer, typename Function>
	inline void for_each(Integer begin, Integer end, Function function) {
	    for (; begin != end; ++begin) {
	        function(begin);
	    }
	}

	template <typename Integer, typename Function, typename Real>
	inline Real accumulate(Integer begin, Integer end, Real sum, Function function) {
		for (; begin != end; ++begin) {
			sum += function(begin);
		}
		return sum;
	}

#ifdef USE_C_ASYNC
	template <typename Integer, typename Function, typename Real>
	inline Real accumulate_thread(Integer begin, Integer end, Real sum, Function function) {
		std::vector<std::future<Real>> results;

		int chunkSize = (end - begin) / nThreads;
		int start = 0;

		for (int i = 0; i < nThreads - 1; ++i, start += chunkSize) {
			results.emplace_back(std::async([=] {
				return accumulate(begin + start, begin + start + chunkSize, 0.0, function);
			}));
		}
		results.emplace_back(std::async([=] {
			return accumulate(begin + start, end, 0.0, function);
		}));

		for (auto&& result : results) {
			sum += result.get();
		}
		return sum;
	}
#endif

#ifdef USE_OMOP
	template <typename Integer, typename Function, typename Real>
	inline Real accumulate_omp(Integer begin, Integer end, Real sum, Function function) {
		#pragma omp
		for (; begin != end; ++begin) {
			sum += function(begin);
		}
		return sum;
	}
#endif

#ifdef USE_THREAD_POOL
	template <typename Integer, typename Function, typename Real>
	inline Real accumulate_thread_pool(Integer begin, Integer end, Real sum, Function function) {
		std::vector<std::future<Real>> results;

		int chunkSize = (end - begin) / nThreads;
		int start = 0;

		for (int i = 0; i < nThreads - 1; ++i, start += chunkSize) {
			results.emplace_back(
				pool.enqueue([=] {
					return accumulate(
						begin + start,
						begin + start + chunkSize,
						Real(0),
						function);
				})
			);
		}
		results.emplace_back(
			pool.enqueue([=] {
				return accumulate(
					begin + start,
					end,
					Real(0),
					function);
			})

		);

		Real total = static_cast<Real>(0);
		for (auto&& result : results) {
			total += result.get();
		}
		return total;
	}
#endif

#ifdef USE_TBB
	template <typename Integer, typename Function, typename Real>
	inline Real accumulate_tbb(Integer begin, Integer end, Real sum, Function function) {
		return tbb::parallel_reduce(
 			tbb::blocked_range<size_t>(begin, end
 			//, 200
 			),
 			sum,
 			[function](const tbb::blocked_range<size_t>& r, Real sum) -> Real {
 				//return accumulate
 				const auto end = r.end();
 				for (auto i = r.begin(); i != end; ++i) {
 					sum += function(i);
 				}
 				return sum;
 			},
 			std::plus<Real>()
		);
	}
#endif


// #include <numeric>
// #include <functional>
// #include "tbb/parallel_reduce.h"
// #include "tbb/blocked_range.h"
//
// using namespace tbb;
//
// float ParallelSum( float array[], size_t n ) {
//     return parallel_reduce(
//         blocked_range<float*>( array, array+n ),
//         0.f,
//         [](const blocked_range<float*>& r, float value)->float {
//             return std::accumulate(r.begin(),r.end(),value);
//         },
//         std::plus<float>()
//     );
// }


// float ParallelSumFoo( const float a[], size_t n ) {
//     SumFoo sf(a);
//     parallel_reduce( blocked_range<size_t>(0,n), sf );
//     return sf.my_sum;
// }
// class SumFoo {
//     float* my_a;
// public:
//     float my_sum;
//     void operator()( const blocked_range<size_t>& r ) {
//         float *a = my_a;
//         float sum = my_sum;
//         size_t end = r.end();
//         for( size_t i=r.begin(); i!=end; ++i )
//             sum += Foo(a[i]);
//         my_sum = sum;
//     }
//
//     SumFoo( SumFoo& x, split ) : my_a(x.my_a), my_sum(0) {}
//
//     void join( const SumFoo& y ) {my_sum+=y.my_sum;}
//
//     SumFoo(float a[] ) :
//         my_a(a), my_sum(0)
//     {}
// };


#endif

private:
	double precision;
	double storedPrecision;

	double oneOverSd;
	double storedOneOverSd;

    double sumOfSquaredResiduals;
    double storedSumOfSquaredResiduals;

    double sumOfTruncations;
    double storedSumOfTruncations;

    mm::MemoryManager<RealType> observations;

    mm::MemoryManager<RealType> locations0;
    mm::MemoryManager<RealType> locations1;

    mm::MemoryManager<RealType>* locationsPtr;
    mm::MemoryManager<RealType>* storedLocationsPtr;

    mm::MemoryManager<RealType> squaredResiduals;
    mm::MemoryManager<RealType> storedSquaredResiduals;

    mm::MemoryManager<RealType> truncations;
    mm::MemoryManager<RealType> storedTruncations;

    bool isStoredSquaredResidualsEmpty;
    bool isStoredTruncationsEmpty;

    bool isStoredAllTruncationsEmpty;

    int nThreads;
//     ThreadPool pool;

};

} // namespace mds

#endif // _ABSTRACTMULTIDIMENSIONALSCALING_HPP
