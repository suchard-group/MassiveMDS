#ifndef _NEWMULTIDIMENSIONALSCALING_HPP
#define _NEWMULTIDIMENSIONALSCALING_HPP

#include <numeric>
#include <vector>

#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include "tbb/global_control.h"

#ifdef RBUILD
#include <Rcpp.h>
#endif

//#define XSIMD_ENABLE_FALLBACK

#include "xsimd/xsimd.hpp"
#include "AbstractMultiDimensionalScaling.hpp"
#include "Distance.hpp"

#if defined(__ARM64_ARCH_8__)
  #undef USE_AVX
  #undef USE_AVX512
#endif

namespace mds {

	struct DoubleNoSimdTypeInfo {
		using BaseType = double;
		using SimdType = double;
		static const int SimdSize = 1;
	};

    struct FloatNoSimdTypeInfo {
        using BaseType = float;
        using SimdType = float;
        static const int SimdSize = 1;
    };

#ifdef USE_SIMD

#ifdef USE_SSE
    struct DoubleSseTypeInfo {
        using BaseType = double;
        using SimdType = xsimd::batch<double, 2>;
        static const int SimdSize = 2;
    };

    struct FloatSseTypeInfo {
        using BaseType = float;
        using SimdType = xsimd::batch<float, 4>;
        static const int SimdSize = 4;
    };
#endif

#ifdef USE_AVX
    struct DoubleAvxTypeInfo {
        using BaseType = double;
        using SimdType = xsimd::batch<double, 4>;
        static const int SimdSize = 4;
    };
#endif

#ifdef USE_AVX512
    struct DoubleAvx512TypeInfo {
        using BaseType = double;
        using SimdType = xsimd::batch<double, 8>;
        static const int SimdSize = 8;
    };
#endif

#endif

template <typename TypeInfo, typename ParallelType>
class NewMultiDimensionalScaling : public AbstractMultiDimensionalScaling {
public:

	using RealType = typename TypeInfo::BaseType;

    NewMultiDimensionalScaling(int embeddingDimension, int locationCount, long flags, int threads, int bandwidth)
        : AbstractMultiDimensionalScaling(embeddingDimension, locationCount, flags, bandwidth),
          precision(0.0), storedPrecision(0.0),
          oneOverSd(0.0), storedOneOverSd(0.0),
          sumOfIncrements(0.0), storedSumOfIncrements(0.0),

          observations(locationCount * locationCount),

          locations0(locationCount * embeddingDimension),
		  locations1(locationCount * embeddingDimension),
		  locationsPtr(&locations0),
		  storedLocationsPtr(&locations1),

          increments(locationCount * bandwidth),

          storedIncrements(locationCount),
          gradientPtr(&gradient0),


          isStoredIncrementsEmpty(false),

          nThreads(threads)
    {

    	if (flags & Flags::LEFT_TRUNCATION) {
    		isLeftTruncated = true;
#ifdef RBUILD
        Rcpp::Rcout << "Using left truncation" << std::endl;
#else
    		std::cout << "Using left truncation" << std::endl;
#endif
    	}

#ifdef USE_TBB
        if (flags & mds::Flags::TBB) {
    		if (nThreads <= 0) {
    		  nThreads = tbb::this_task_arena::max_concurrency();
    		}
#ifdef RBUILD
    		    Rcpp::Rcout << "Using " << nThreads << " threads" << std::endl;
#else
            std::cout << "Using " << nThreads << " threads" << std::endl;
#endif

            control = std::make_shared<tbb::global_control>(tbb::global_control::max_allowed_parallelism, nThreads);
    	}
#endif
    }



    virtual ~NewMultiDimensionalScaling() { }

	int getInternalDimension() { return embeddingDimension; }

    void updateLocations(int locationIndex, double* location, size_t length) {

		size_t offset{0};

		if (locationIndex == -1) {
			// Update all locations
			assert(length == embeddingDimension * locationCount);

			incrementsKnown = false;
			isStoredIncrementsEmpty = true;

			// TODO Do anything with updatedLocation?
		} else {
			// Update a single location
    		assert(length == embeddingDimension);

	    	if (updatedLocation != - 1) {
    			// more than one location updated -- do a full recomputation
	    		incrementsKnown = false;
	    		isStoredIncrementsEmpty = true;
    		}

	    	updatedLocation = locationIndex;
	    	offset = locationIndex * embeddingDimension;
	    }

		mm::bufferedCopy(location, location + length,
			begin(*locationsPtr) + offset,
			buffer
		);

    	sumOfIncrementsKnown = false;
    }

    void computeIncrements() {
		if (!incrementsKnown) {
			if (isLeftTruncated) { // run-time dispatch to compile-time optimization
				if (embeddingDimension == 2) {
					computeSumOfIncrementsGeneric<true, typename TypeInfo::SimdType, TypeInfo::SimdSize, NonGeneric>();
				} else {
					computeSumOfIncrementsGeneric<true, typename TypeInfo::SimdType, TypeInfo::SimdSize, Generic>();
				}
			} else {
				if (embeddingDimension == 2) {
					computeSumOfIncrementsGeneric<false, typename TypeInfo::SimdType, TypeInfo::SimdSize, NonGeneric>();
				} else {
					computeSumOfIncrementsGeneric<false, typename TypeInfo::SimdType, TypeInfo::SimdSize, Generic>();
				}
			}
			incrementsKnown = true;
		} else {
			if (isLeftTruncated) {
				updateSumOfIncrements<true>();
			} else {
				updateSumOfIncrements<false>();
			}
		}
    }

    double getSumOfIncrements() {
    	if (!sumOfIncrementsKnown) {
			computeIncrements();
			sumOfIncrementsKnown = true;
		}
		if (isLeftTruncated) {
			return sumOfIncrements;
		} else {
			return 0.5 * precision * sumOfIncrements;
		}
 	}

    void storeState() {
    	storedSumOfIncrements = sumOfIncrements;

    	std::copy(begin(*locationsPtr), end(*locationsPtr),
    		begin(*storedLocationsPtr));

    	isStoredIncrementsEmpty = true;

    	updatedLocation = -1;

    	storedPrecision = precision;
    	storedOneOverSd = oneOverSd;
    }

    double getDiagnostic() {
        return std::accumulate(
            begin(increments),
            end(increments),
            RealType(0));
    }

    void acceptState() {
        if (!isStoredIncrementsEmpty) {
    		for (int j = 0; j < locationCount; ++j) {
    			increments[j * locationCount + updatedLocation] = increments[updatedLocation * locationCount + j];
    		}
    	}
    }

    void restoreState() {
    	sumOfIncrements = storedSumOfIncrements;
    	sumOfIncrementsKnown = true;

		if (!isStoredIncrementsEmpty) {
    		std::copy(
    			begin(storedIncrements),
    			end(storedIncrements),
    			begin(increments) + updatedLocation * locationCount
    		);
    		incrementsKnown = true;
    	} else {
    		incrementsKnown = false; // Force recompute;  TODO cache
    	}

    	precision = storedPrecision;
    	oneOverSd = storedOneOverSd;

    	auto tmp1 = storedLocationsPtr;
    	storedLocationsPtr = locationsPtr;
    	locationsPtr = tmp1;
    }

    void setPairwiseData(double* data, size_t length) {
		assert(length == observations.size());
		mm::bufferedCopy(data, data + length, begin(observations), buffer);
    }

    void setParameters(double* data, size_t length) {
		assert(length == 1); // Call only with precision
		precision = data[0]; // TODO Remove
		oneOverSd = std::sqrt(data[0]);

		// Handle truncations
		if (isLeftTruncated) {
			incrementsKnown = false;
			sumOfIncrementsKnown = false;

    		isStoredIncrementsEmpty = true;
		}
    }

    void makeDirty() {
    	sumOfIncrementsKnown = false;
    	incrementsKnown = false;
    }

	void getLogLikelihoodGradient(double* result, size_t length) {

		assert (length == locationCount * embeddingDimension);

        // TODO Cache values

		if (isLeftTruncated) { // run-time dispatch to compile-time optimization
			if (embeddingDimension == 2) {
				computeLogLikelihoodGradientGeneric<true, typename TypeInfo::SimdType, TypeInfo::SimdSize, NonGeneric>();
			} else {
				computeLogLikelihoodGradientGeneric<true, typename TypeInfo::SimdType, TypeInfo::SimdSize, Generic>();
			}
		} else {
			if (embeddingDimension == 2) {
				computeLogLikelihoodGradientGeneric<false, typename TypeInfo::SimdType, TypeInfo::SimdSize, NonGeneric>();
			} else {
				computeLogLikelihoodGradientGeneric<false, typename TypeInfo::SimdType, TypeInfo::SimdSize, Generic>();
			}
		}

		mm::bufferedCopy(std::begin(*gradientPtr), std::end(*gradientPtr), result, buffer);
    }

	template <bool withTruncation, typename SimdType, int SimdSize, typename Algorithm>
    void computeLogLikelihoodGradientGeneric() {

        const auto length = locationCount * embeddingDimension;
        if (length != gradientPtr->size()) {
            gradientPtr->resize(length);
        }

        std::fill(std::begin(*gradientPtr), std::end(*gradientPtr),
                  static_cast<RealType>(0.0));

        //const auto dim = embeddingDimension;
        //RealType* gradient = gradientPtr->data();
        const RealType scale = precision;

        for_each(0, locationCount, [this, scale](const int i) { // [gradient,dim]

          int upperMin = bandwidth + 1;

          if (locationCount - i <= upperMin) {
            upperMin = locationCount-i;
          }

          const int upper = upperMin - upperMin % SimdSize;


          int lowerMin = -(bandwidth + 1);

          if (-i >= lowerMin) {
            lowerMin = -i;
          }

          const int lower = lowerMin - lowerMin % SimdSize;

			DistanceDispatch<SimdType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension);

			innerGradientLoop<withTruncation, SimdType, SimdSize>(dispatch, scale, i, lower, upper);

			// if (vectorCount < locationCount) { // Edge-cases
			//
			// 	DistanceDispatch<RealType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension);
			//
			// 	innerGradientLoop<withTruncation, RealType, 1>(dispatch, scale, i, vectorCount, locationCount);
			// }
        }, ParallelType());
    }

#ifdef USE_SIMD

	template <typename T, size_t N>
	using SimdBatch = xsimd::batch<T, N>;

	template <typename T, size_t N>
	using SimdBatchBool = xsimd::batch_bool<T, N>;

	template <typename T, size_t N>
	bool any(SimdBatchBool<T, N> x) {
	    return xsimd::any(x);
	}

	template <typename T, size_t N>
	T reduce(SimdBatch<T, N> x) {
	    return xsimd::hadd(x);
	}

	template <typename T, size_t N>
	SimdBatch<T, N> mask(SimdBatchBool<T, N> flag, SimdBatch<T, N> x) {
	    return SimdBatch<T, N>(flag()) & x;
	}

	template <typename T, size_t N>
	SimdBatchBool<T, N> getMissing(int i, int j, SimdBatch<T, N> x) {
        return SimdBatch<T, N>(i) == (SimdBatch<T, N>(j) + getIota(SimdBatch<T, N>())) || xsimd::isnan(x);
	}
#endif

#ifdef USE_SSE
	using D2 = xsimd::batch<double, 2>;
	using D2Bool = xsimd::batch_bool<double, 2>;
	using S4 = xsimd::batch<float, 4>;
	using S4Bool = xsimd::batch_bool<float, 4>;

    D2 getIota(D2) {
        return D2(0, 1);
    }

    S4 getIota(S4) {
        return S4(0, 1, 2, 3);
    }
#endif

#ifdef USE_AVX
	using D4 = xsimd::batch<double, 4>;
	using D4Bool = xsimd::batch_bool<double, 4>;

	D4 getIota(D4) {
        return D4(0, 1, 2, 3);
    }
#endif

#ifdef USE_AVX512
    using D8 = xsimd::batch<double, 8>;
	using D8Bool = xsimd::batch_bool<double, 8>;

	D8 getIota(D8) {
	    return D8(0, 1, 2, 3, 4, 5, 6, 7);
	}

	D8 mask(D8Bool flag, D8 x) {
		return xsimd::select(flag, x, D8(0.0)); // bitwise & does not appear to work
    }
#endif

    template <typename T>
    bool getMissing(int i, int j, T x) {
        return i == j || std::isnan(x);
    }

    template <typename T>
    T mask(bool flag, T x) {
        return flag ? x : T(0.0);
    }

    template <typename T>
    T reduce(T x) {
        return x;
    }

	bool any(bool x) {
		return x;
	}

	template <bool withTruncation, typename SimdType, int SimdSize, typename DispatchType>
	void innerGradientLoop(const DispatchType& dispatch, const RealType scale, const int i,
								 const int begin, const int end) {

        const SimdType sqrtScale(std::sqrt(scale));

		for (int j = begin; j < end; j += SimdSize) {

			const auto distance = dispatch.calculate(i+j);
			const auto observation = SimdHelper<SimdType, RealType>::get(&observations[i * locationCount + i + j]);
			const auto notMissing = !getMissing(i, i+j, observation);

			if (any(notMissing)) {

				auto residual = mask(notMissing, observation - distance);

				if (withTruncation) {

					residual -= mask(notMissing, math::pdf_new( distance * sqrtScale ) /
									  (xsimd::exp(math::phi_new(distance * sqrtScale)) *
									   sqrtScale) );
				}

				auto dataContribution = mask(notMissing, residual * scale / distance);

                for (int k = 0; k < SimdSize; ++k) {
                    for (int d = 0; d < embeddingDimension; ++d) {

                        const RealType something = getScalar(dataContribution, k);

                        const RealType update = something *
                                                ((*locationsPtr)[i * embeddingDimension + d] -
                                                 (*locationsPtr)[(i + j + k) * embeddingDimension + d]);


                        (*gradientPtr)[i * embeddingDimension + d] += update;
                    }
                }
			}
		}
	}

	double getScalar(double x, int i) {
		return x;
	}

	float getScalar(float x, int i) {
		return x;
	}

#ifdef USE_SIMD
#ifdef USE_AVX
	double getScalar(D4 x, int i) {
		return x[i];
	}
#endif
#ifdef USE_SSE
	double getScalar(D2 x, int i) {
		return x[i];
	}

	float getScalar(S4 x, int i) {
		return x[i];
	}
#endif
#endif // USE_SIMD

#ifdef USE_AVX512
    double getScalar(D8 x, int i) {
	    return x[i];
	}
#endif

    template <bool withTruncation, typename SimdType, int SimdSize, typename DispatchType>
    RealType innerLikelihoodLoop(const DispatchType& dispatch, const RealType scale, const int i,
                                 const int begin, const int end) {

        SimdType sum = SimdType(RealType(0));

        for (int j = begin; j < end; j += SimdSize) {

            const auto distance = dispatch.calculate(i+j);
            const auto observation = SimdHelper<SimdType, RealType>::get(&observations[i * locationCount + i + j]);
            const auto notMissing = !getMissing(i, i+j, observation);

            if (any(notMissing)) {

                const auto residual = mask(notMissing, observation - distance);
                auto squaredResidual = residual * residual;

                if (withTruncation) {
                    squaredResidual *= scale;
                    squaredResidual += mask(notMissing, math::phi_new(distance * oneOverSd));
                }

                SimdHelper<SimdType, RealType>::put(squaredResidual, &increments[i * bandwidth + j]);
                sum += squaredResidual;
            }
        }

        return reduce(sum);
    }

    template <bool withTruncation, typename SimdType, int SimdSize, typename Algorithm>
    void computeSumOfIncrementsGeneric() {

        const auto scale = 0.5 * precision;

        RealType delta =
                accumulate(0, locationCount, RealType(0), [this, scale](const int i) {
                   int bandMin = bandwidth + 1;

                   if (locationCount - i <= bandMin) {
                      bandMin = locationCount-i;
                   }


                    const int vectorCount = bandMin - bandMin % SimdSize;


                    //Rcpp::Rcout << "bandMin: " << bandMin << std::endl;
                    //
                   //  Rcpp::Rcout << "VectorCount: " << vectorCount << std::endl;


                    DistanceDispatch<SimdType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension);

					RealType sumOfSquaredResiduals =
                            innerLikelihoodLoop<withTruncation, SimdType, SimdSize>(dispatch, scale, i,
                                                                                    0, vectorCount);


                   // if (vectorCount < bandMin) { // Edge-cases
                   //
                   //       DistanceDispatch<RealType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension);
                   //
                   //       sumOfSquaredResiduals +=
                   //               innerLikelihoodLoop<withTruncation, RealType, 1>(dispatch, scale, i,
                   //                                                                vectorCount, bandMin);
                   //   }

                    return sumOfSquaredResiduals;

                }, ParallelType());

        double lSumOfSquaredResiduals = delta;

        lSumOfSquaredResiduals /= 2.0;
        sumOfIncrements = lSumOfSquaredResiduals;

        incrementsKnown = true;
        sumOfIncrementsKnown = true;
    }

	template <bool withTruncation>
	void updateSumOfIncrements() { // TODO To be vectorized (when we start using this function again)

		assert (false); // Should not get here anymore
		const RealType scale = RealType(0.5) * precision;

		const int i = updatedLocation;
		isStoredIncrementsEmpty = false;

		//auto start  = begin(*locationsPtr) + i * embeddingDimension;
		//auto offset = begin(*locationsPtr);

		RealType delta =
 		accumulate(0, locationCount, RealType(0),

			[this, i, // , &offset, &start]
      scale](const int j) {
                const auto distance = 0.0; // TODO
//                        calculateDistance<mm::MemoryManager<RealType>>(
//                    start,
//                    offset + embeddingDimension * j,
//                    embeddingDimension
//                );

                const auto observation = observations[i * locationCount + j];
                auto squaredResidual = RealType(0);

                if (!std::isnan(observation)) {
                    const auto residual = distance - observation;
                    squaredResidual = residual * residual;

                    if (withTruncation) { // Compile-time
                        squaredResidual = scale * squaredResidual;
                        if (i != j) {
                            squaredResidual += math::phi2<NewMultiDimensionalScaling>(distance * oneOverSd);
                        }
                    }
                }

            	// store old value
            	const auto oldSquaredResidual = increments[i * locationCount + j];
            	storedIncrements[j] = oldSquaredResidual;

                const auto inc = squaredResidual - oldSquaredResidual;

            	// store new value
                increments[i * locationCount + j] = squaredResidual;

                return inc;
            }, ParallelType()
		);

		sumOfIncrements += delta;
	}

// Parallelization helper functions

	template <typename Integer, typename Function>
	inline void for_each(Integer begin, Integer end, Function function, CpuAccumulate) {
	    for (; begin != end; ++begin) {
	        function(begin);
	    }
	}

	template <typename Integer, typename Function, typename Real>
	inline Real accumulate(Integer begin, Integer end, Real sum, Function function, CpuAccumulate) {
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
	inline Real accumulate(Integer begin, Integer end, Real sum, Function function, TbbAccumulate) {

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

	template <typename Integer, typename Function>
	inline void for_each(Integer begin, const Integer end, Function function, TbbAccumulate) {

		tbb::parallel_for(
				tbb::blocked_range<size_t>(begin, end
				//, 200
				),
				[function](const tbb::blocked_range<size_t>& r) -> void {
					const auto end = r.end();
					for (auto i = r.begin(); i != end; ++i) {
						function(i);
					}
				}
		);
	}
#endif

private:
	double precision;
	double storedPrecision;

	double oneOverSd;
	double storedOneOverSd;

    double sumOfIncrements;
    double storedSumOfIncrements;

    mm::MemoryManager<RealType> observations;

    mm::MemoryManager<RealType> locations0;
    mm::MemoryManager<RealType> locations1;

    mm::MemoryManager<RealType>* locationsPtr;
    mm::MemoryManager<RealType>* storedLocationsPtr;

    mm::MemoryManager<RealType> increments;
    mm::MemoryManager<RealType> storedIncrements;

    mm::MemoryManager<RealType> gradient0;
    mm::MemoryManager<RealType> gradient1;

    mm::MemoryManager<RealType>* gradientPtr;
    mm::MemoryManager<RealType>* storedGradientPtr;

    mm::MemoryManager<double> buffer;

    bool isStoredIncrementsEmpty;

    int nThreads;

#ifdef USE_TBB
    std::shared_ptr<tbb::global_control> control;
#endif

};

// factory
std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingDoubleNoParallelNoSimd(int embeddingDimension, int locationCount, long flags, int threads, int bandwidth) {
#ifdef RBUILD
	Rcpp::Rcout << "DOUBLE, NO PARALLEL, NO SIMD" << std::endl;
#else
  std::cerr << "DOUBLE, NO PARALLEL, NO SIMD" << std::endl;
#endif
	return std::make_shared<NewMultiDimensionalScaling<DoubleNoSimdTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads, bandwidth);
}

std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingDoubleTbbNoSimd(int embeddingDimension, int locationCount, long flags, int threads, int bandwidth) {
#ifdef RBUILD
  Rcpp::Rcout << "DOUBLE, TBB PARALLEL, NO SIMD" << std::endl;
#else
  std::cerr << "DOUBLE, TBB PARALLEL, NO SIMD" << std::endl;
#endif
  return std::make_shared<NewMultiDimensionalScaling<DoubleNoSimdTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads, bandwidth);
}

std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingFloatNoParallelNoSimd(int embeddingDimension, int locationCount, long flags, int threads, int bandwidth) {
#ifdef RBUILD
  Rcpp::Rcout << "SINGLE, NO PARALLEL, NO SIMD" << std::endl;
#else
  std::cerr << "SINGLE, NO PARALLEL, NO SIMD" << std::endl;
#endif
  return std::make_shared<NewMultiDimensionalScaling<FloatNoSimdTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads, bandwidth);
}

std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingFloatTbbNoSimd(int embeddingDimension, int locationCount, long flags, int threads, int bandwidth) {
#ifdef RBUILD
  Rcpp::Rcout << "SINGLE, TBB PARALLEL, NO SIMD" << std::endl;
#else
  std::cerr << "SINGLE, TBB PARALLEL, NO SIMD" << std::endl;
#endif
  return std::make_shared<NewMultiDimensionalScaling<FloatNoSimdTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads, bandwidth);
}

#ifdef USE_SIMD

#ifdef USE_AVX
    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleNoParallelAvx(int embeddingDimension, int locationCount, long flags, int threads, int bandwidth) {
#ifdef RBUILD
        Rcpp::Rcout << "DOUBLE, NO PARALLEL, AVX" << std::endl;
#else
        std::cerr << "DOUBLE, NO PARALLEL, AVX" << std::endl;
#endif
        return std::make_shared<NewMultiDimensionalScaling<DoubleAvxTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads, bandwidth);
    }

    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleTbbAvx(int embeddingDimension, int locationCount, long flags, int threads, int bandwidth) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, TBB PARALLEL, AVX" << std::endl;
#else
      std::cerr << "DOUBLE, TBB PARALLEL, AVX" << std::endl;
#endif
        return std::make_shared<NewMultiDimensionalScaling<DoubleAvxTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads, bandwidth);
    }
#endif

#ifdef USE_AVX512
    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleNoParallelAvx512(int embeddingDimension, int locationCount, long flags, int threads, int bandwidth) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, NO PARALLEL, AVX512" << std::endl;
#else
      std::cerr << "DOUBLE, NO PARALLEL, AVX512" << std::endl;
#endif
        return std::make_shared<NewMultiDimensionalScaling<DoubleAvx512TypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads, bandwidth);
    }

    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleTbbAvx512(int embeddingDimension, int locationCount, long flags, int threads, int bandwidth) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, TBB PARALLEL, AVX512" << std::endl;
#else
      std::cerr << "DOUBLE, TBB PARALLEL, AVX512" << std::endl;
#endif
        return std::make_shared<NewMultiDimensionalScaling<DoubleAvx512TypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads, bandwidth);
    }
#endif

#ifdef USE_SSE
    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleNoParallelSse(int embeddingDimension, int locationCount, long flags, int threads, int bandwidth) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, NO PARALLEL, SSE" << std::endl;
#else
      std::cerr << "DOUBLE, NO PARALLEL, SSE" << std::endl;
#endif
      return std::make_shared<NewMultiDimensionalScaling<DoubleSseTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads, bandwidth);
    }

    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleTbbSse(int embeddingDimension, int locationCount, long flags, int threads, int bandwidth) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, TBB PARALLEL, SSE" << std::endl;
#else
      std::cerr << "DOUBLE, TBB PARALLEL, SSE" << std::endl;
#endif
      return std::make_shared<NewMultiDimensionalScaling<DoubleSseTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads, bandwidth);
    }

	std::shared_ptr<AbstractMultiDimensionalScaling>
	constructNewMultiDimensionalScalingFloatNoParallelSse(int embeddingDimension, int locationCount, long flags, int threads, int bandwidth) {
#ifdef RBUILD
	  Rcpp::Rcout << "SINGLE, NO PARALLEL, SSE" << std::endl;
#else
	  std::cerr << "SINGLE, NO PARALLEL, SSE" << std::endl;
#endif
	  return std::make_shared<NewMultiDimensionalScaling<FloatSseTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads, bandwidth);
	}

	std::shared_ptr<AbstractMultiDimensionalScaling>
	constructNewMultiDimensionalScalingFloatTbbSse(int embeddingDimension, int locationCount, long flags, int threads, int bandwidth) {
#ifdef RBUILD
	  Rcpp::Rcout << "SINGLE, TBB PARALLEL, SSE" << std::endl;
#else
	  std::cerr << "SINGLE, TBB PARALLEL, SSE" << std::endl;
#endif
	  return std::make_shared<NewMultiDimensionalScaling<FloatSseTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads, bandwidth);
	}
#endif

#endif

} // namespace mds

#endif // _NEWMULTIDIMENSIONALSCALING_HPP
