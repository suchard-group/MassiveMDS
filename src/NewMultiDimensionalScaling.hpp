#ifndef _NEWMULTIDIMENSIONALSCALING_HPP
#define _NEWMULTIDIMENSIONALSCALING_HPP

#include <numeric>
#include <vector>
#include <tbb/task_scheduler_init.h>

//#define XSIMD_ENABLE_FALLBACK

#include "xsimd/xsimd.hpp"
#include "AbstractMultiDimensionalScaling.hpp"
#include "Distance.hpp"


//#undef SSE
//#define SSE

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

    struct FloatSimdTypeInfo {
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

    NewMultiDimensionalScaling(int embeddingDimension, int locationCount, long flags, int threads)
        : AbstractMultiDimensionalScaling(embeddingDimension, locationCount, flags),
          precision(0.0), storedPrecision(0.0),
          oneOverSd(0.0), storedOneOverSd(0.0),
          sumOfIncrements(0.0), storedSumOfIncrements(0.0),

          observations(locationCount * locationCount),

          locations0(locationCount * embeddingDimension),
		  locations1(locationCount * embeddingDimension),
		  locationsPtr(&locations0),
		  storedLocationsPtr(&locations1),

		  gradientPtr(&gradient0),

          increments(locationCount * locationCount),
          storedIncrements(locationCount),

          isStoredIncrementsEmpty(false)

          , nThreads(4) //, pool(nThreads)
    {

    	if (flags & Flags::LEFT_TRUNCATION) {
    		isLeftTruncated = true;
    		std::cout << "Using left truncation" << std::endl;
    	}

        if (flags & mds::Flags::TBB) {
    		if (threads==0) {
    			threads = tbb::task_scheduler_init::default_num_threads();
    		}
            std::cout << "Using " << threads << " threads" << std::endl;
			std::make_shared<tbb::task_scheduler_init>(threads);
    	}
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

        const auto dim = embeddingDimension;
        RealType* gradient = gradientPtr->data();
        const RealType scale = precision;

        for_each(0, locationCount, [this, gradient, scale, dim](const int i) {

			const int vectorCount = locationCount - locationCount % SimdSize;

			DistanceDispatch<SimdType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension);

			innerGradientLoop<withTruncation, SimdType, SimdSize>(dispatch, scale, i, 0, vectorCount);
//			SimdHelper<SimdType, RealType>::put(innerGradientLoop<withTruncation, SimdType, SimdSize>(dispatch,
//																									  scale, i, 0,
//																									  vectorCount),
//												&gradient[i * embeddingDimension]);



			if (vectorCount < locationCount) { // Edge-cases

				DistanceDispatch<RealType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension);

				//gradient[i * embeddingDimension] +=
				innerGradientLoop<withTruncation, RealType, 1>(dispatch, scale, i, vectorCount, locationCount);
//				SimdHelper<SimdType, RealType>::put(innerGradientLoop<withTruncation, RealType, 1>(dispatch,
//																								   scale, i,
//																								   vectorCount,
//																								   locationCount),
//													&gradient[i * embeddingDimension]);
			}
        }, ParallelType());
    };

    template <bool withTruncation>
    void computeLogLikelihoodGradientNew() {

        assert (embeddingDimension == 2);

		const auto length = locationCount * embeddingDimension;
		if (length != gradientPtr->size()) {
			gradientPtr->resize(length);
		}

		RealType* gradient = gradientPtr->data();
		const RealType scale = precision;

		// TODO This is easy to parallelize, but runs a bit slower than the old version
		for_each(0, locationCount, [this, gradient, scale](const int i) {

			// TODO Use SIMD
			RealType gradij0 = 0.0;
			RealType gradij1 = 0.0;

			for (int j = 0; j < locationCount; ++j) {
				if (i != j) {
					const auto distance = calculateDistance<mm::MemoryManager<RealType>>(
						begin(*locationsPtr) + i * embeddingDimension,
						begin(*locationsPtr) + j * embeddingDimension,
						embeddingDimension
					);

                    const RealType observation = observations[i * locationCount + j];
					RealType residual = std::isnan(observation) ?
										RealType(0) :
										observation - distance;

					if (withTruncation) {
						const RealType trncDrv = std::isnan(observation) ?
												 RealType(0) :
												 -pdf(distance * sqrt(scale)) /
												 (std::exp(math::phi2<NewMultiDimensionalScaling>(distance * sqrt(scale))) * sqrt(scale));
						residual = residual + trncDrv;
					}

					const RealType dataContribution = residual * scale / distance;

					const RealType update0 = dataContribution *
						((*locationsPtr)[i * embeddingDimension + 0] - (*locationsPtr)[j * embeddingDimension + 0]);
					const RealType update1 = dataContribution *
						((*locationsPtr)[i * embeddingDimension + 1] - (*locationsPtr)[j * embeddingDimension + 1]);

                    if (withTruncation) {
                        // TODO
                    }

					gradij0 += update0;
					gradij1 += update1;
				}
			}

			gradient[i * embeddingDimension + 0] = gradij0;
			gradient[i * embeddingDimension + 1] = gradij1;

		}, ParallelType());
	};

    template <bool withTruncation>
	void computeLogLikelihoodGradientOld() {

		const auto length = locationCount * embeddingDimension;
		if (length != gradientPtr->size()) {
			gradientPtr->resize(length);
		}

		std::fill(std::begin(*gradientPtr), std::end(*gradientPtr), static_cast<RealType>(0.0));

		RealType* gradient = gradientPtr->data();

		const RealType scale = precision;

		for (int i = 0; i < locationCount; ++i) {

			for (int j = i; j < locationCount; ++j) {
				if (i != j) {
					const auto distance = calculateDistance<mm::MemoryManager<RealType>>(
							begin(*locationsPtr) + i * embeddingDimension,
							begin(*locationsPtr) + j * embeddingDimension,
							embeddingDimension
					);

                    const RealType observation = observations[i * locationCount + j];
					const RealType dataContribution = std::isnan(observation) ?
                                                      RealType(0) :
                                                      (observation - distance) * scale / distance;

					const RealType update0 = dataContribution *
											 ((*locationsPtr)[i * embeddingDimension + 0] - (*locationsPtr)[j * embeddingDimension + 0]);
					const RealType update1 = dataContribution *
											 ((*locationsPtr)[i * embeddingDimension + 1] - (*locationsPtr)[j * embeddingDimension + 1]);

                    if (withTruncation) {
                        // TODO
                    }

					gradient[i * embeddingDimension + 0] += update0;
					gradient[i * embeddingDimension + 1] += update1;

					gradient[j * embeddingDimension + 0] -= update0;
					gradient[j * embeddingDimension + 1] -= update1;

				}
			}
		}
	};

	int count = 0;

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
//
//	using D4 = xsimd::batch<double, 4>;
//	using D4Bool = xsimd::batch_bool<double, 4>;
//
//	using D2 = xsimd::batch<double, 2>;
//	using D2Bool = xsimd::batch_bool<double, 2>;
//
//	using S4 = xsimd::batch<float, 4>;
//	using S4Bool = xsimd::batch_bool<float, 4>;
//
//    D4 getIota(D4) {
//        return D4(0, 1, 2, 3);
//    }
//
//    D2 getIota(D2) {
//        return D2(0, 1);
//    }
//
//    S4 getIota(S4) {
//        return S4(0, 1, 2, 3);
//    }
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

			const auto distance = dispatch.calculate(j);
			const auto observation = SimdHelper<SimdType, RealType>::get(&observations[i * locationCount + j]);
			const auto notMissing = !getMissing(i, j, observation);

			if (any(notMissing)) {

				auto residual = mask(notMissing, observation - distance);

				if (withTruncation) {

					residual += mask(notMissing, math::pdf_new( distance * sqrtScale ) /
									  (xsimd::exp(math::phi_new(distance * sqrtScale)) *
									   sqrtScale) );
				}

				auto dataContribution = mask(notMissing, residual * scale / distance);

                for (int k = 0; k < SimdSize; ++k) {
                    for (int d = 0; d < embeddingDimension; ++d) {

                        const RealType something = getScalar(dataContribution, k);

                        const RealType update = something *
                                                ((*locationsPtr)[i * embeddingDimension + d] -
                                                 (*locationsPtr)[(j + k) * embeddingDimension + d]);


                        (*gradientPtr)[i * embeddingDimension + d] += update;
                    }
                }
			}
		}
	};

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

            const auto distance = dispatch.calculate(j);
            const auto observation = SimdHelper<SimdType, RealType>::get(&observations[i * locationCount + j]);
            const auto notMissing = !getMissing(i, j, observation);

            if (any(notMissing)) {

                const auto residual = mask(notMissing, observation - distance);
                auto squaredResidual = residual * residual;

                if (withTruncation) {
                    squaredResidual *= scale;
                    squaredResidual += mask(notMissing, math::phi_new(distance * oneOverSd));
                }

                SimdHelper<SimdType, RealType>::put(squaredResidual, &increments[i * locationCount + j]);
                sum += squaredResidual;
            }
        }

        return reduce(sum);
    };

    template <bool withTruncation, typename SimdType, int SimdSize, typename Algorithm>
    void computeSumOfIncrementsGeneric() {

        const auto scale = 0.5 * precision;

        RealType delta =
                accumulate(0, locationCount, RealType(0), [this, scale](const int i) {

                    const int vectorCount = locationCount - locationCount % SimdSize;

                    DistanceDispatch<SimdType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension);

					RealType sumOfSquaredResiduals =
                            innerLikelihoodLoop<withTruncation, SimdType, SimdSize>(dispatch, scale, i,
                                                                                    0, vectorCount);


                    if (vectorCount < locationCount) { // Edge-cases

                        DistanceDispatch<RealType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension);

                        sumOfSquaredResiduals +=
                                innerLikelihoodLoop<withTruncation, RealType, 1>(dispatch, scale, i,
                                                                                 vectorCount, locationCount);
                    }

                    return sumOfSquaredResiduals;

                }, ParallelType());

        double lSumOfSquaredResiduals = delta;

        lSumOfSquaredResiduals /= 2.0;
        sumOfIncrements = lSumOfSquaredResiduals;

        incrementsKnown = true;
        sumOfIncrementsKnown = true;
    }

//	template <bool withTruncation>
//	void computeSumOfIncrements() {
//
//    	assert(false);
//        assert (embeddingDimension == 2);
//
//		const RealType scale = 0.5 * precision;
//
//		RealType delta =
//		accumulate(0, locationCount, RealType(0), [this, scale](const int i) {
//
//			RealType lSumOfSquaredResiduals{0};
//
//			for (int j = 0; j < locationCount; ++j) {
//
//				const auto distance = calculateDistance<mm::MemoryManager<RealType>>(
//					begin(*locationsPtr) + i * embeddingDimension,
//					begin(*locationsPtr) + j * embeddingDimension,
//					embeddingDimension
//				);
//
//				const auto observation = observations[i * locationCount + j];
//                auto squaredResidual = RealType(0);
//
//                if (!std::isnan(observation)) {
//
//                    const auto residual = distance - observation;
//
//                    squaredResidual = residual * residual;
//
//                    if (withTruncation) {
//                        squaredResidual = scale * squaredResidual;
//                        if (i != j) {
//                            squaredResidual += math::phi2<NewMultiDimensionalScaling>(distance * oneOverSd);
//                        }
//                    }
//                }
//
//				increments[i * locationCount + j] = squaredResidual;
//				lSumOfSquaredResiduals += squaredResidual;
//
//			}
//			return lSumOfSquaredResiduals;
//		}, ParallelType());
//
//		double lSumOfSquaredResiduals = delta;
//
//    	lSumOfSquaredResiduals /= 2.0;
//    	sumOfIncrements = lSumOfSquaredResiduals;
//
//	    incrementsKnown = true;
//	    sumOfIncrementsKnown = true;
//	}

	template <bool withTruncation>
	void updateSumOfIncrements() { // TODO To be vectorized (when we start using this function again)

		assert (false); // Should not get here anymore
		const RealType scale = RealType(0.5) * precision;

		const int i = updatedLocation;
		isStoredIncrementsEmpty = false;

		auto start  = begin(*locationsPtr) + i * embeddingDimension;
		auto offset = begin(*locationsPtr);

		RealType delta =
 		accumulate(0, locationCount, RealType(0),

			[this, i, &offset, scale,
			&start](const int j) {
                const auto distance = calculateDistance<mm::MemoryManager<RealType>>(
                    start,
                    offset + embeddingDimension * j,
                    embeddingDimension
                );

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


#ifdef SSE

	template <typename VectorType, typename Iterator, typename SimdType>
	RealType calculateDistanceXsimd(Iterator iX, Iterator iY, int length) const {

		SimdType sum{0.0};

		for (int i = 0; i < length; i += SimdType::size) {
            const auto diff = SimdType(iX + i, xsimd::aligned_mode()) -
                              SimdType(iY + i, xsimd::aligned_mode());
            sum += diff * diff;
		}


		return std::sqrt(sum.hadd());
	}

    template <typename VectorType, typename Iterator>
    RealType calculateDistanceGeneric(Iterator iX, Iterator iY, int length) const {

        RealType sum = static_cast<RealType>(0.0);

        for (int i = 0; i < length; ++i, ++iX, ++iY) {
            const auto diff = *iX - *iY;
            sum += diff * diff;
        }

        return std::sqrt(sum);
    }

	#ifdef __clang__

    template <typename VectorType, typename Iterator>
    RealType calculateDistance(Iterator iX, Iterator iY, int length) const {
    	return calculateDistance(iX, iY, length, RealType());
    }

    template <typename Iterator>
    RealType calculateDistance(Iterator iX, Iterator iY, int length, float) const {

        //using AlignedValueType = typename HostVectorType::allocator_type::aligned_value_type;

        typedef float aligned_float __attribute__((aligned(16)));
        typedef aligned_float* SSE_PTR;

        SSE_PTR __restrict__ x = &*iX;
        SSE_PTR __restrict__ y = &*iY;

        auto a = _mm_loadu_ps(x);
        auto b = _mm_loadu_ps(y); // TODO second call is not aligned without padding
        auto c = a - b;

        const int mask = 0x31;
        __m128 d = _mm_dp_ps(c, c, mask);
        return  _mm_cvtss_f32(_mm_sqrt_ps(d));
    }

//   template <typename Iterator>
//    RealType calculateDistance(Iterator iX, Iterator iY, int length, float) const {
//
//        using AlignedValueType = typename mm::MemoryManager<float>::allocator_type::aligned_value_type;
//
//        AlignedValueType* x = &*iX;
//        AlignedValueType* y = &*iY;
//
//        auto sum = static_cast<AlignedValueType>(0);
//        for (int i = 0; i < 2; ++i, ++x, ++y) {
//            const auto difference = *x - *y;
//            sum += difference * difference;
//        }
//
//        return std::sqrt(sum);
//    }

   template <typename Iterator>
    RealType calculateDistance(Iterator iX, Iterator iY, int length, double) const {

        using AlignedValueType = typename mm::MemoryManager<double>::allocator_type::aligned_value_type;

        AlignedValueType* x = &*iX;
        AlignedValueType* y = &*iY;

        __m128d xv = _mm_load_pd(x);
        __m128d yv = _mm_load_pd(y);

       __m128d diff = _mm_sub_pd(xv, yv);

	   const int mask = 0x31;
	   __m128d d = _mm_dp_pd(diff, diff, mask);
	   return  std::sqrt(_mm_cvtsd_f64(d));
    }

	#else // __clang__

  template <typename VectorType, typename Iterator>
  RealType calculateDistance(Iterator iX, Iterator iY, int length) const {
 	return calculateDistance(iX, iY, length, RealType());
  }

  //namespace detail {
   template <typename Iterator>
    RealType calculateDistance(Iterator iX, Iterator iY, int length, float) const {

        //using AlignedValueType = typename HostVectorType::allocator_type::aligned_value_type;

 	typedef float aligned_float __attribute__((aligned(16)));
  	typedef aligned_float* SSE_PTR;

	SSE_PTR __restrict__ x = &*iX;
	SSE_PTR __restrict__ y = &*iY;

	auto a = _mm_loadu_ps(x);
	auto b = _mm_loadu_ps(y); // TODO second call is not aligned without padding
	auto c = a - b;

	const int mask = 0x31;
	__m128 d = _mm_dp_ps(c, c, mask);
	return  _mm_cvtss_f32(_mm_sqrt_ps(d));
    }

   template <typename Iterator>
   RealType calculateDistance(Iterator iX, Iterator iY, int length, double) const {

        //using AlignedValueType = typename HostVectorType::allocator_type::aligned_value_type;

 	typedef double aligned_double __attribute__((aligned(16)));
  	typedef aligned_double* SSE_PTR;

	SSE_PTR __restrict__ x = &*iX;
	SSE_PTR __restrict__ y = &*iY;

	const auto a = _mm_load_pd(x);
	const auto b = _mm_load_pd(y);
	const auto c = a - b;

	const int mask = 0x31;
	__m128d d = _mm_dp_pd(c, c, mask);
	return  _mm_cvtsd_f64(_mm_sqrt_pd(d));
    }
	//} // namespace detail

    #endif // __clang__

#else // SSE

	// Non-SIMD implementations

    template <typename VectorType, typename Iterator>
    double calculateDistance(Iterator x, Iterator y, int length) const {
        auto sum = static_cast<double>(0);

        for (int i = 0; i < 2; ++i, ++x, ++y) {
            const auto difference = *x - *y;
            sum += difference * difference;
        }
        return std::sqrt(sum);
    }

	template <typename VectorType, typename Iterator>
	double calculateDistanceGeneric(Iterator x, Iterator y, int length) const {
		auto sum = static_cast<double>(0);

		for (int i = 0; i < length; ++i, ++x, ++y) {
			const auto difference = *x - *y;
			sum += difference * difference;
		}
		return std::sqrt(sum);
	}
#endif // SSE

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
	};
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

};

// factory
std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingDoubleNoParallelNoSimd(int embeddingDimension, int locationCount, long flags, int threads) {
	std::cerr << "DOUBLE, NO PARALLEL, NO SIMD" << std::endl;
	return std::make_shared<NewMultiDimensionalScaling<DoubleNoSimdTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
}

std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingDoubleTbbNoSimd(int embeddingDimension, int locationCount, long flags, int threads) {
	std::cerr << "DOUBLE, TBB PARALLEL, NO SIMD" << std::endl;
	return std::make_shared<NewMultiDimensionalScaling<DoubleNoSimdTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
}

std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingFloatNoParallelNoSimd(int embeddingDimension, int locationCount, long flags, int threads) {
	std::cerr << "SINGLE, NO PARALLEL, NO SIMD" << std::endl;
    return std::make_shared<NewMultiDimensionalScaling<FloatNoSimdTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
}

std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingFloatTbbNoSimd(int embeddingDimension, int locationCount, long flags, int threads) {
	std::cerr << "SINGLE, TBB PARALLEL, NO SIMD" << std::endl;
    return std::make_shared<NewMultiDimensionalScaling<FloatNoSimdTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
}

#ifdef USE_SIMD

#ifdef USE_AVX
    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleNoParallelAvx(int embeddingDimension, int locationCount, long flags, int threads) {
        std::cerr << "DOUBLE, NO PARALLEL, AVX" << std::endl;
        return std::make_shared<NewMultiDimensionalScaling<DoubleAvxTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
    }

    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleTbbAvx(int embeddingDimension, int locationCount, long flags, int threads) {
        std::cerr << "DOUBLE, TBB PARALLEL, AVX" << std::endl;
        return std::make_shared<NewMultiDimensionalScaling<DoubleAvxTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
    }
#endif

#ifdef USE_AVX512
    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleNoParallelAvx512(int embeddingDimension, int locationCount, long flags, int threads) {
        std::cerr << "DOUBLE, NO PARALLEL, AVX512" << std::endl;
        return std::make_shared<NewMultiDimensionalScaling<DoubleAvx512TypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
    }

    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleTbbAvx512(int embeddingDimension, int locationCount, long flags, int threads) {
        std::cerr << "DOUBLE, TBB PARALLEL, AVX512" << std::endl;
        return std::make_shared<NewMultiDimensionalScaling<DoubleAvx512TypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
    }
#endif

#ifdef USE_SSE
    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleNoParallelSse(int embeddingDimension, int locationCount, long flags, int threads) {
        std::cerr << "DOUBLE, NO PARALLEL, SSE" << std::endl;
        return std::make_shared<NewMultiDimensionalScaling<DoubleSseTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
    }

    std::shared_ptr<AbstractMultiDimensionalScaling>
    constructNewMultiDimensionalScalingDoubleTbbSse(int embeddingDimension, int locationCount, long flags, int threads) {
        std::cerr << "DOUBLE, TBB PARALLEL, SSE" << std::endl;
        return std::make_shared<NewMultiDimensionalScaling<DoubleSseTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
    }

	std::shared_ptr<AbstractMultiDimensionalScaling>
	constructNewMultiDimensionalScalingFloatNoParallelSse(int embeddingDimension, int locationCount, long flags, int threads) {
		std::cerr << "SINGLE, NO PARALLEL, SSE" << std::endl;
		return std::make_shared<NewMultiDimensionalScaling<FloatSimdTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
	}

	std::shared_ptr<AbstractMultiDimensionalScaling>
	constructNewMultiDimensionalScalingFloatTbbSse(int embeddingDimension, int locationCount, long flags, int threads) {
		std::cerr << "SINGLE, TBB PARALLEL, SSE" << std::endl;
		return std::make_shared<NewMultiDimensionalScaling<FloatSimdTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
	}
#endif

#endif

} // namespace mds

#endif // _NEWMULTIDIMENSIONALSCALING_HPP
