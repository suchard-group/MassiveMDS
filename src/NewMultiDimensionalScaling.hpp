#ifndef _NEWMULTIDIMENSIONALSCALING_HPP
#define _NEWMULTIDIMENSIONALSCALING_HPP

#include <numeric>
#include <vector>

#define XSIMD_ENABLE_FALLBACK

#include "xsimd/xsimd.hpp"
#include "AbstractMultiDimensionalScaling.hpp"
#include "Distance.hpp"

//#undef SSE
#define SSE

namespace mds {

template <typename RealType, typename ParallelType>
class NewMultiDimensionalScaling : public AbstractMultiDimensionalScaling {
public:
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

	#define SIMD_TYPE xsimd::batch<RealType, 2>
	#define SIMD_SIZE 2

// 	#define SIMD_TYPE xsimd::batch<RealType, 1>
// 	#define SIMD_SIZE 1

// 	#define SIMD_TYPE double
// 	#define SIMD_SIZE 1

       using D2 = xsimd::batch<RealType, 2>;
// 		using SimdType = double;

		if (!incrementsKnown) {
			if (isLeftTruncated) { // run-time dispatch to compile-time optimization
				if (embeddingDimension == 2) {
					computeSumOfIncrementsGeneric<true, SIMD_TYPE, SIMD_SIZE, NonGeneric>();
				} else {
					computeSumOfIncrementsGeneric<true, SIMD_TYPE, SIMD_SIZE, Generic>();
				}
			} else {
				if (embeddingDimension == 2) {
					computeSumOfIncrementsGeneric<false, SIMD_TYPE, SIMD_SIZE, NonGeneric>();
				} else {
					computeSumOfIncrementsGeneric<false, SIMD_TYPE, SIMD_SIZE, Generic>();
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

	double pdf(double value) { // standard normal density
		return 0.398942280401432677939946059934 * std::exp( - value * value * 0.5);
	}

    void makeDirty() {
    	sumOfIncrementsKnown = false;
    	incrementsKnown = false;
    }

	void getLogLikelihoodGradient(double* result, size_t length) {

		assert (length == locationCount * embeddingDimension);

        // TODO Cache values

#define TEST_PARALLEL

#ifdef TEST_PARALLEL
        if (isLeftTruncated) { // run-time dispatch to compile-time optimization
            if (embeddingDimension == 2) {
                computeLogLikelihoodGradientNew<true>();
            } else {
                computeLogLikelihoodGradientGeneric<true>();
            }
        } else {
            if (embeddingDimension == 2) {
                computeLogLikelihoodGradientNew<false>();
            } else {
                computeLogLikelihoodGradientGeneric<false>();
            }
        }
#else
        if (isLeftTruncated) { // run-time dispatch to compile-time optimization
            computeLogLikelihoodGradientOld<true>();
        } else {
            computeLogLikelihoodGradientOld<false>();
        }
#endif // TEST_PARALLEL

		mm::bufferedCopy(std::begin(*gradientPtr), std::end(*gradientPtr), result, buffer);
    }

    template <bool withTruncation>
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

            for (int j = 0; j < locationCount-1; j+=2) {

            	RealType distances [2];
            	distances[0] = calculateDistanceGeneric<mm::MemoryManager<RealType>>(
						begin(*locationsPtr) + i * dim,
						begin(*locationsPtr) + j * dim,
						dim
				);
				distances[1] = calculateDistanceGeneric<mm::MemoryManager<RealType>>(
						begin(*locationsPtr) + i * dim,
						begin(*locationsPtr) + (j + 1) * dim,
						dim
				);

				using b_type = xsimd::simd_type<RealType>;
				b_type distance;
				xsimd::load_unaligned(&distances[0],distance);

				b_type neqI;
				RealType bool1 [2];
				bool1[0] = i!=j ? 1.0 : RealType(0);
				bool1[1] = i!=(j+1) ? 1.0 : RealType(0);
				xsimd::load_unaligned( &bool1[0], neqI );

				const auto observation1 = observations[i * locationCount + j];
				const auto observation2 = observations[i * locationCount + (j+1)];

				b_type nNan;
				RealType bool2 [2];
				bool2[0] = std::isnan(observation1) ? RealType(0): 1.0;
				bool2[1] = std::isnan(observation2) ? RealType(0): 1.0;
				xsimd::load_unaligned( &bool2[0], nNan );

				RealType residuals [2];

				residuals[0] = (std::isnan(observation1) ? RealType(0) : observation1 - distances[0]) *
									   (i != j);
				residuals[1] = (std::isnan(observation2) ? RealType(0) : observation2 - distances[1]) *
									   (i != (j + 1));
				b_type residual;
				xsimd::load_unaligned( &residuals[0] , residual );

				if (withTruncation) {
					RealType zero [2];
					zero[0] = RealType(0);
					zero[1] = RealType(0);
					b_type zeros;
					b_type trncDrv;

					xsimd::load_unaligned(&zero[0],zeros);

					trncDrv = zeros + math::pdf_new( distance * sqrt(scale) ) /
									  (xsimd::exp(math::phi_new(distance * sqrt(scale))) *
									   sqrt(scale)) * nNan * neqI;
//					RealType jnt_trnc [2] = math::pdf_new( distance * sqrt(scale) ) /
//										(xsimd::exp(math::phi_new(distance * sqrt(scale))) *
//										 sqrt(scale));
//					b_type trncDrv;
//					xsimd::load_unaligned( &jnt_trnc[0], trncDrv );


//					jnt_trnc[0] = std::isnan(observation1) || i == j ? RealType(0) : trncDrv[0];
//					jnt_trnc[1] = std::isnan(observation2) || i == (j+1) ? RealType(0) : trncDrv[1];
//					xsimd::load_unaligned(&jnt_trnc[0],trncDrv);

					residual += trncDrv;
				}


				RealType dataContribution1 = std::isnan(observation1) || (i == j) ?
											RealType(0) :
											residual[0] * scale / distance[0];
				RealType dataContribution2 = std::isnan(observation2) || (i == (j+1)) ?
											 RealType(0) :
											 residual[1] * scale / distance[1];

				for (int d = 0; d < dim; ++d) {
					const RealType update1 = dataContribution1 *
											((*locationsPtr)[i * dim + d] - (*locationsPtr)[j * dim + d]);
					const RealType update2 = dataContribution2 *
											((*locationsPtr)[i * dim + d] - (*locationsPtr)[(j+1) * dim + d]);


					gradient[i * dim + d] += update1 + update2;
				}
            }
			if (locationCount % 2 != 0 && i != (locationCount - 1)) { // one more time for individual j = location count - 1

            	int j = locationCount - 1;

				const auto distance = calculateDistanceGeneric<mm::MemoryManager<RealType>>(
						begin(*locationsPtr) + i * dim,
						begin(*locationsPtr) + j * dim,
						dim
				);

				const auto observation = observations[i * locationCount + j];
				RealType residual = (std::isnan(observation) ? RealType(0) : observation - distance);


				if (withTruncation) {
					const RealType trncDrv = std::isnan(observation) ?
											  RealType(0) :
											  pdf(distance * sqrt(scale)) /
											  (std::exp(math::phi2<NewMultiDimensionalScaling>(distance * sqrt(scale))) *
											   sqrt(scale));
					residual = residual + trncDrv;
				}

				RealType dataContribution = std::isnan(observation) ?
											 RealType(0) :
											 residual * scale / distance;

				for (int d = 0; d < dim; ++d) {
					const RealType update = dataContribution *
											 ((*locationsPtr)[i * dim + d] - (*locationsPtr)[j * dim + d]);

					gradient[i * dim + d] += update;
				}
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

//#define TEST_GENERIC_GRADIENT

#ifdef TEST_GENERIC_GRADIENT
        auto twoSum = std::accumulate(std::begin(*gradientPtr), std::end(*gradientPtr), 0.0);

        std::fill(std::begin(*gradientPtr), std::end(*gradientPtr), 1.0);

        computeLogLikelihoodGradientGeneric<withTruncation>();
        auto genericSum = std::accumulate(std::begin(*gradientPtr), std::end(*gradientPtr), 0.0);
        std::cerr << genericSum << " " << twoSum << std::endl;
        exit(0);
#endif // TEST_GENERIC_GRADIENT

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

	using D2 = xsimd::batch<double, 2>;
	using D2Bool = xsimd::batch_bool<double, 2>;

	using D1 = xsimd::batch<double, 1>;
	using D1Bool = xsimd::batch_bool<double, 1>;

	D2Bool getMissing(int i, int j, D2 x) {
		return D2Bool(i == j, i == j + 1) || xsimd::isnan(x);
	}

	D1Bool getMissing(int i, int j, D1 x) {
		return D1Bool(i == j) || xsimd::isnan(x());
	}

	bool getMissing(int i, int j, double x) {
		return i == j || std::isnan(x);
	}

//	D2 makeMask(D2Bool x) {
//		return xsimd::select(x, D2(0.0, 0.0), D2(1.0, 1.0));
//	}
//
//	D1 makeMask(D1Bool x) {
//		return xsimd::select(x, D1(0.0), D1(1.0));
//	}
//
//	double makeMask(bool x) {
//		return x ? 0.0 : 1.0;
//	}

    D2 mask(D2Bool flag, D2 x) {
        return D2(flag()) & x;
    }

//    D1 mask(D1Bool flag, D1 x) {
//        return D1(flag.size) & x; // TODO Fix
//    }

    double mask(bool flag, double x) {
        return flag ? x : 0.0;
    }

	bool any(D2Bool x) {
		return xsimd::any(x);
	}

	bool any(D1Bool x) {
		return xsimd::any(x);
	}

	bool any(bool x) {
		return x;
	}

    bool all(D2Bool x) {
        return xsimd::all(x);
    }

    bool all(D1Bool x) {
        return xsimd::all(x);
    }

    bool all(bool x) {
        return x;
    }

	double reduce(D2 x) {
		return xsimd::hadd(x);
	}

	double reduce(D1 x) {
		return xsimd::hadd(x);
	}

	double reduce(double x) {
		return x;
	}

    template <bool withTruncation, typename SimdType, typename DispatchType>
    SimdType incrementLikelihood(const DispatchType& dispatch, const RealType scale, const int i, const int j) {

        const auto distance = dispatch.calculate(j);
        const auto observation = SimdHelper<SimdType, RealType>::get(&observations[i * locationCount + j]);
        const auto notMissing = !getMissing(i, j, observation);

        auto squaredResidual = SimdType(0.0);
        if (any(notMissing)) {

            const auto residual = mask(notMissing, observation - distance);
            squaredResidual = residual * residual;

            if (withTruncation) {
                squaredResidual *= scale;
                squaredResidual += mask(notMissing, math::phi_new(distance * oneOverSd));
            }

            SimdHelper<SimdType, RealType>::put(squaredResidual, &increments[i * locationCount + j]);
        }

        return squaredResidual;
    };

    template <bool withTruncation, typename SimdType, int SimdSize, typename Algorithm>
    void computeSumOfIncrementsGeneric() {

        const int vectorCount = locationCount - locationCount % SimdSize;

        const auto scale = 0.5 * precision;

        RealType delta =
                accumulate(0, locationCount, RealType(0), [this, scale](const int i) {

                    auto vSumOfSquaredResiduals = SimdType(0.0);

                    DistanceDispatch<SimdType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension);

                    const int vectorCount = locationCount - locationCount % SimdSize;
                    
                    for (int j = 0; j < vectorCount; j += SimdSize) {
                        vSumOfSquaredResiduals += incrementLikelihood<withTruncation, SimdType>(dispatch, scale, i, j);
                    } // end for

					RealType sumOfSquaredResiduals = reduce(vSumOfSquaredResiduals);

                    if (vectorCount < locationCount) { // Edge-cases

                        DistanceDispatch<RealType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension);

                        for (int j = vectorCount; j < locationCount; ++j) {
                            sumOfSquaredResiduals += incrementLikelihood<withTruncation, RealType>(dispatch, scale, i, j);
                        }
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

// 	int count = 0
	int count2 = 0;

	template <bool withTruncation>
	void updateSumOfIncrements() {

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

//	double phi2(double value) const {
//		return log(0.5 * erfc(-value * M_SQRT1_2));
//	}

    using b_type = xsimd::simd_type<RealType>;
    //b_type = xsimd::batch<RealType,2>;
	b_type phi2(b_type scaledDistance, b_type neqI, b_type nNan) const {
		scaledDistance *= -M_SQRT1_2;
		scaledDistance = xsimd::erfc(scaledDistance);
		scaledDistance *= 0.5;
		scaledDistance = xsimd::log(scaledDistance);
		scaledDistance *= neqI * nNan;
		return scaledDistance;
	}
//
//	__m128d phi2(__m128d value) const {
//		const __m128d scalar = _mm_set1_ps(-M_SQRT1_2);
//		value = _mm_mul_ps(value, scalar);
//		return xsimd::log(0.5 * xsimd::erfc(value));
//	}

#ifdef SSE

    void fun_test() {

        mm::MemoryManager<RealType> x(10);
        mm::MemoryManager<RealType> y(10);

        RealType result2 = calculateDistanceXsimd<xsimd::batch<RealType, 2>>(
                std::begin(x), std::begin(y), 10
        );

        RealType result1 = calculateDistanceXsimd<xsimd::batch<RealType, 1>>(
                std::begin(x), std::begin(y), 10
        );
    }

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
constructNewMultiDimensionalScalingDoubleNoParallel(int embeddingDimension, int locationCount, long flags, int threads) {
	std::cerr << "DOUBLE, NO PARALLEL" << std::endl;
	return std::make_shared<NewMultiDimensionalScaling<double, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
}

std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingDoubleTbb(int embeddingDimension, int locationCount, long flags, int threads) {
	std::cerr << "DOUBLE, TBB PARALLEL" << std::endl;
	return std::make_shared<NewMultiDimensionalScaling<double, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
}

std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingFloatNoParallel(int embeddingDimension, int locationCount, long flags, int threads) {
	std::cerr << "SINGLE, NO PARALLEL" << std::endl;
//	return std::make_shared<NewMultiDimensionalScaling<float, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
    return std::make_shared<NewMultiDimensionalScaling<double, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
}

std::shared_ptr<AbstractMultiDimensionalScaling>
constructNewMultiDimensionalScalingFloatTbb(int embeddingDimension, int locationCount, long flags, int threads) {
	std::cerr << "SINGLE, TBB PARALLEL" << std::endl;
//	return std::make_shared<NewMultiDimensionalScaling<float, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
    return std::make_shared<NewMultiDimensionalScaling<double, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
}

} // namespace mds

#endif // _NEWMULTIDIMENSIONALSCALING_HPP
