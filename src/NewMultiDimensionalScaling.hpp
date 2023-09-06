#ifndef _NEWMULTIDIMENSIONALSCALING_HPP
#define _NEWMULTIDIMENSIONALSCALING_HPP

#include <numeric>
#include <vector>

#define TBB_PREVIEW_GLOBAL_CONTROL 1
#ifdef USE_TBB
	#include "tbb/global_control.h"
#endif

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

    NewMultiDimensionalScaling(int embeddingDimension, Layout layout, long flags, int threads)
        : AbstractMultiDimensionalScaling(embeddingDimension, layout, flags),
          precision(0.0), storedPrecision(0.0),
          oneOverSd(0.0), storedOneOverSd(0.0),
          sumOfIncrements(0.0), storedSumOfIncrements(0.0),

          observations(layout.observationCount),
          transposedObservations(layout.isSymmetric() ? 0 : layout.observationCount),

          locations0(layout.uniqueLocationCount * embeddingDimension),
		  locations1(layout.uniqueLocationCount * embeddingDimension),
		  locationsPtr(&locations0),
		  storedLocationsPtr(&locations1),

          increments(layout.observationCount),

          storedIncrements(layout.observationCount),
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
			assert(length == embeddingDimension * layout.uniqueLocationCount);

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
//    	std::cerr << "sumOfInc = " << sumOfIncrements << "\n";
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
            const int count = layout.uniqueLocationCount;
    		for (int j = 0; j < count; ++j) {
    			increments[j * count + updatedLocation] = increments[updatedLocation * count + j];
    		}
    	}
    }

    void restoreState() {
    	sumOfIncrements = storedSumOfIncrements;
    	sumOfIncrementsKnown = true;

		if (!isStoredIncrementsEmpty) {
		    const int count = layout.uniqueLocationCount;
    		std::copy(
    			begin(storedIncrements),
    			end(storedIncrements),
    			begin(increments) + updatedLocation * count
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

    //template <typename T>
    //void maskOutDiagonal()

    void setPairwiseData(double* data, size_t length) {
		assert(length == observations.size());
		mm::bufferedCopy(data, data + length, begin(observations), buffer);

		if (layout.isSymmetric()) {
		  for (int i = 0; i < layout.rowLocationCount; ++i) {
		    observations[i * layout.observationStride + i] = std::nan("");
		  }
		} else {
		  for (int i = 0; i < layout.rowLocationCount; ++i) {
		    for (int j = 0; j < layout.columnLocationCount; ++j) {
		      // TODO benchmark, then lazy update or SIMD
		      transposedObservations[j * layout.rowLocationCount + i] = observations[i * layout.columnLocationCount +j];
		    }
		  }
		}

		makeDirty();
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

		const auto count = layout.uniqueLocationCount;

		assert (length == count * embeddingDimension);

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

		const auto count = layout.uniqueLocationCount;
        const auto length = count * embeddingDimension;
        if (length != gradientPtr->size()) {
            gradientPtr->resize(length);
        }

        std::fill(std::begin(*gradientPtr), std::end(*gradientPtr),
                  static_cast<RealType>(0.0));

        const RealType scale = precision;

        const auto rowLocationCount = layout.rowLocationCount;
        const auto columnLocationCount = layout.columnLocationCount;
        const auto columnLocationOffset = layout.columnLocationOffset;

        for_each(0, rowLocationCount,
                 [this, scale, rowLocationCount, columnLocationCount, columnLocationOffset](const int i) {

                   const int vectorCount = columnLocationCount - columnLocationCount % SimdSize;

                   DistanceDispatch<SimdType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension, 0, columnLocationOffset);

                   innerGradientLoop<withTruncation, SimdType, SimdSize>(dispatch, observations, layout.observationStride, scale, i, 0, vectorCount);

                   if (vectorCount < columnLocationCount) { // Edge-cases

                     DistanceDispatch<RealType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension,0, columnLocationOffset);

                     innerGradientLoop<withTruncation, RealType, 1>(dispatch, observations, layout.observationStride, scale, i, vectorCount, columnLocationCount);
                   }
                 }, ParallelType());

        for_each(0, columnLocationCount,
                 [this, scale, rowLocationCount, columnLocationCount, columnLocationOffset](const int i) {

                   const int vectorCount = rowLocationCount - rowLocationCount % SimdSize;

                   DistanceDispatch<SimdType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension, columnLocationOffset, 0); // ABCD

                   innerGradientLoop<withTruncation, SimdType, SimdSize>(dispatch, transposedObservations, layout.transposedObservationStride, scale, i, 0, vectorCount);

                   if (vectorCount < rowLocationCount) { // Edge-cases

                     DistanceDispatch<RealType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension, columnLocationOffset, 0);

                     innerGradientLoop<withTruncation, RealType, 1>(dispatch, transposedObservations, layout.transposedObservationStride, scale, i, vectorCount, rowLocationCount);
                   }
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
        // return SimdBatch<T, N>(i) == (SimdBatch<T, N>(j) + getIota(SimdBatch<T, N>())) || xsimd::isnan(x);
	      return xsimd::isnan(x);
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
        // return i == j || std::isnan(x);
        return std::isnan(x);
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
	void innerGradientLoop(const DispatchType& dispatch,
                        mm::MemoryManager<RealType>& observations, const int stride,
                        const RealType scale, const int i,
								 const int begin, const int end) {

        const SimdType sqrtScale(std::sqrt(scale));

        //const auto stride = layout.observationStride; // TODO UPDATE

		for (int j = begin; j < end; j += SimdSize) {

			const auto distance = dispatch.calculate(j);
			const auto observation = SimdHelper<SimdType, RealType>::get(&observations[i * stride + j]);
			const auto notMissing = !getMissing(i, j, observation);

			if (any(notMissing)) {

				auto residual = mask(notMissing, observation - distance);

				if (withTruncation) {

					residual -= mask(notMissing, math::pdf_new( distance * sqrtScale ) /
									  (xsimd::exp(math::phi_new(distance * sqrtScale)) *
									   sqrtScale) );
				}

				auto dataContribution = mask(notMissing, residual * scale / distance);
				const auto rowLocationOffset = dispatch.getRowLocationOffset();
				const auto columnLocationOffset = dispatch.getColumnLocationOffset();
				// const auto rowOffset = dispatch.getRowOffset();

                for (int k = 0; k < SimdSize; ++k) {
                    for (int d = 0; d < embeddingDimension; ++d) {

                        const RealType something = getScalar(dataContribution, k);

                        const RealType update = something *
                                                ((*locationsPtr)[rowLocationOffset + i * embeddingDimension + d] -
                                                 (*locationsPtr)[columnLocationOffset + (j + k) * embeddingDimension + d]);


                        (*gradientPtr)[rowLocationOffset + i * embeddingDimension + d] += update;
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

		    const auto stride = layout.observationStride;

        SimdType sum = SimdType(RealType(0));

        for (int j = begin; j < end; j += SimdSize) {

            const auto distance = dispatch.calculate(j);
            const auto observation = SimdHelper<SimdType, RealType>::get(&observations[i * stride + j]);
            const auto notMissing = !getMissing(i, j, observation);

            if (any(notMissing)) {

                const auto residual = mask(notMissing, observation - distance);
                auto squaredResidual = residual * residual;

                if (withTruncation) {
                    squaredResidual *= scale;
                    squaredResidual += mask(notMissing, math::phi_new(distance * oneOverSd));
                }

                SimdHelper<SimdType, RealType>::put(squaredResidual, &increments[i * stride + j]);
                sum += squaredResidual;
            }
        }

        return reduce(sum);
    }

    template <bool withTruncation, typename SimdType, int SimdSize, typename Algorithm>
    void computeSumOfIncrementsGeneric() {

       const auto scale = 0.5 * precision;

        RealType delta =
                accumulate(0, layout.rowLocationCount, RealType(0), [this, scale](const int i) {

                    const auto rowLocationCount = layout.rowLocationCount;
                    const auto columnLocationOffset = layout.columnLocationOffset;
                    const auto columnLocationCount = layout.columnLocationCount;

                    const int vectorCount = columnLocationCount - columnLocationCount % SimdSize;

                    DistanceDispatch<SimdType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension, 0, columnLocationOffset);

					RealType sumOfSquaredResiduals =
                            innerLikelihoodLoop<withTruncation, SimdType, SimdSize>(dispatch, scale, i,
                                                                                    0, vectorCount);


                    if (vectorCount < columnLocationCount) { // Edge-cases

                        DistanceDispatch<RealType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension, 0, columnLocationOffset);

                        sumOfSquaredResiduals +=
                                innerLikelihoodLoop<withTruncation, RealType, 1>(dispatch, scale, i,
                                                                                 vectorCount, columnLocationCount);
                    }

                    return sumOfSquaredResiduals;

                }, ParallelType());

        double lSumOfSquaredResiduals = delta;

        if (layout.isSymmetric()) {
          lSumOfSquaredResiduals /= 2.0;
        }
        sumOfIncrements = lSumOfSquaredResiduals;

        incrementsKnown = true;
        sumOfIncrementsKnown = true;
    }

	template <bool withTruncation>
	void updateSumOfIncrements() { // TODO To be vectorized (when we start using this function again)

		const auto stride = layout.observationStride; // TODO UPDATE

		assert (false); // Should not get here anymore
		const RealType scale = RealType(0.5) * precision;

		const int i = updatedLocation;
		isStoredIncrementsEmpty = false;

		//auto start  = begin(*locationsPtr) + i * embeddingDimension;
		//auto offset = begin(*locationsPtr);

		RealType delta =
 		accumulate(0, layout.rowLocationCount, RealType(0),

			[this, i, stride, // , &offset, &start]
      scale](const int j) {
                const auto distance = 0.0; // TODO
//                        calculateDistance<mm::MemoryManager<RealType>>(
//                    start,
//                    offset + embeddingDimension * j,
//                    embeddingDimension
//                );

                const auto observation = observations[i * stride + j];
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
            	const auto oldSquaredResidual = increments[i * stride + j];
            	storedIncrements[j] = oldSquaredResidual;

                const auto inc = squaredResidual - oldSquaredResidual;

            	// store new value
                increments[i * stride + j] = squaredResidual;

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
    mm::MemoryManager<RealType> transposedObservations;

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
constructNewMultiDimensionalScalingDoubleNoParallelNoSimd(int embeddingDimension, Layout layout, long flags, int threads) {
#ifdef RBUILD
	Rcpp::Rcout << "DOUBLE, NO PARALLEL, NO SIMD" << std::endl;
#else
  std::cerr << "DOUBLE, NO PARALLEL, NO SIMD" << std::endl;
#endif
	return std::make_shared<NewMultiDimensionalScaling<DoubleNoSimdTypeInfo, CpuAccumulate>>(embeddingDimension, layout, flags, threads);
}

#ifdef USE_TBB
std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingDoubleTbbNoSimd(int embeddingDimension, Layout layout, long flags, int threads) {
#ifdef RBUILD
  Rcpp::Rcout << "DOUBLE, TBB PARALLEL, NO SIMD" << std::endl;
#else
  std::cerr << "DOUBLE, TBB PARALLEL, NO SIMD" << std::endl;
#endif
  return std::make_shared<NewMultiDimensionalScaling<DoubleNoSimdTypeInfo, TbbAccumulate>>(embeddingDimension, layout, flags, threads);
}
#endif

std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingFloatNoParallelNoSimd(int embeddingDimension, Layout layout, long flags, int threads) {
#ifdef RBUILD
  Rcpp::Rcout << "SINGLE, NO PARALLEL, NO SIMD" << std::endl;
#else
  std::cerr << "SINGLE, NO PARALLEL, NO SIMD" << std::endl;
#endif
  return std::make_shared<NewMultiDimensionalScaling<FloatNoSimdTypeInfo, CpuAccumulate>>(embeddingDimension, layout, flags, threads);
}

#ifdef USE_TBB
std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingFloatTbbNoSimd(int embeddingDimension, Layout layout, long flags, int threads) {
#ifdef RBUILD
  Rcpp::Rcout << "SINGLE, TBB PARALLEL, NO SIMD" << std::endl;
#else
  std::cerr << "SINGLE, TBB PARALLEL, NO SIMD" << std::endl;
#endif
  return std::make_shared<NewMultiDimensionalScaling<FloatNoSimdTypeInfo, TbbAccumulate>>(embeddingDimension, layout, flags, threads);
}
#endif

#ifdef USE_SIMD

#ifdef USE_AVX
    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleNoParallelAvx(int embeddingDimension, Layout layout, long flags, int threads) {
#ifdef RBUILD
        Rcpp::Rcout << "DOUBLE, NO PARALLEL, AVX" << std::endl;
#else
        std::cerr << "DOUBLE, NO PARALLEL, AVX" << std::endl;
#endif
        return std::make_shared<NewMultiDimensionalScaling<DoubleAvxTypeInfo, CpuAccumulate>>(embeddingDimension, layout, flags, threads);
    }

#ifdef USE_TBB
    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleTbbAvx(int embeddingDimension, Layout layout, long flags, int threads) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, TBB PARALLEL, AVX" << std::endl;
#else
      std::cerr << "DOUBLE, TBB PARALLEL, AVX" << std::endl;
#endif
        return std::make_shared<NewMultiDimensionalScaling<DoubleAvxTypeInfo, TbbAccumulate>>(embeddingDimension, layout, flags, threads);
    }
#endif
#endif

#ifdef USE_AVX512
    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleNoParallelAvx512(int embeddingDimension, Layout layout, long flags, int threads) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, NO PARALLEL, AVX512" << std::endl;
#else
      std::cerr << "DOUBLE, NO PARALLEL, AVX512" << std::endl;
#endif
        return std::make_shared<NewMultiDimensionalScaling<DoubleAvx512TypeInfo, CpuAccumulate>>(embeddingDimension, layout, flags, threads);
    }

#ifdef USE_TBB
    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleTbbAvx512(int embeddingDimension, Layout layout, long flags, int threads) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, TBB PARALLEL, AVX512" << std::endl;
#else
      std::cerr << "DOUBLE, TBB PARALLEL, AVX512" << std::endl;
#endif
        return std::make_shared<NewMultiDimensionalScaling<DoubleAvx512TypeInfo, TbbAccumulate>>(embeddingDimension, layout, flags, threads);
    }
#endif
#endif

#ifdef USE_SSE
    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleNoParallelSse(int embeddingDimension, Layout layout, long flags, int threads) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, NO PARALLEL, SSE" << std::endl;
#else
      std::cerr << "DOUBLE, NO PARALLEL, SSE" << std::endl;
#endif
      return std::make_shared<NewMultiDimensionalScaling<DoubleSseTypeInfo, CpuAccumulate>>(embeddingDimension, layout, flags, threads);
    }

#ifdef USE_TBB
    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleTbbSse(int embeddingDimension, Layout layout, long flags, int threads) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, TBB PARALLEL, SSE" << std::endl;
#else
      std::cerr << "DOUBLE, TBB PARALLEL, SSE" << std::endl;
#endif
      return std::make_shared<NewMultiDimensionalScaling<DoubleSseTypeInfo, TbbAccumulate>>(embeddingDimension, layout, flags, threads);
    }
#endif

	std::shared_ptr<AbstractMultiDimensionalScaling>
	constructNewMultiDimensionalScalingFloatNoParallelSse(int embeddingDimension, Layout layout, long flags, int threads) {
#ifdef RBUILD
	  Rcpp::Rcout << "SINGLE, NO PARALLEL, SSE" << std::endl;
#else
	  std::cerr << "SINGLE, NO PARALLEL, SSE" << std::endl;
#endif
	  return std::make_shared<NewMultiDimensionalScaling<FloatSseTypeInfo, CpuAccumulate>>(embeddingDimension, layout, flags, threads);
	}

#ifdef USE_TBB
	std::shared_ptr<AbstractMultiDimensionalScaling>
	constructNewMultiDimensionalScalingFloatTbbSse(int embeddingDimension, Layout layout, long flags, int threads) {
#ifdef RBUILD
	  Rcpp::Rcout << "SINGLE, TBB PARALLEL, SSE" << std::endl;
#else
	  std::cerr << "SINGLE, TBB PARALLEL, SSE" << std::endl;
#endif
	  return std::make_shared<NewMultiDimensionalScaling<FloatSseTypeInfo, TbbAccumulate>>(embeddingDimension, layout, flags, threads);
	}
#endif
#endif

#endif

} // namespace mds

#endif // _NEWMULTIDIMENSIONALSCALING_HPP
