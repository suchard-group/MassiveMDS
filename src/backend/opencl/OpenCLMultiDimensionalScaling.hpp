#ifndef _OPENCLMULTIDIMENSIONALSCALING_HPP
#define _OPENCLMULTIDIMENSIONALSCALING_HPP

#include "AbstractMultiDimensionalScaling.hpp"

#define SSE

#include "OpenCLMemoryManagement.hpp"

namespace mds {

template <typename RealType>
class OpenCLMultiDimensionalScaling : public AbstractMultiDimensionalScaling {
public:
    OpenCLMultiDimensionalScaling(int embeddingDimension, int locationCount, long flags)
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
          isStoredTruncationsEmpty(false)//,
//          isStoredAllTruncationsEmpty(false)

//           , nThreads(4) //, pool(nThreads)
    {


		device = boost::compute::system::default_device();
		std::cerr << device.name() << std::endl;


    	if (flags & Flags::LEFT_TRUNCATION) {
    		isLeftTruncated = true;
    		std::cout << "Using left truncation" << std::endl;

    		truncations.resize(locationCount * locationCount);
    		storedTruncations.resize(locationCount);
    	}

    	std::cerr << "ctor OpenCLMultiDimensionalScaling" << std::endl;
    }

    virtual ~OpenCLMultiDimensionalScaling() { }

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
				updateSumOfSquaredResidualsAndTruncations();
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

//     double getDiagnostic() {
//         return std::accumulate(
//             begin(squaredResiduals),
//             end(squaredResiduals),
//             RealType(0));
//     }

    void acceptState() {
        if (!isStoredSquaredResidualsEmpty) {
    		for (int j = 0; j < locationCount; ++j) {
    			squaredResiduals[j * locationCount + updatedLocation] = squaredResiduals[updatedLocation * locationCount + j];
    		}
    		if (isLeftTruncated) {
                for (int j = 0; j < locationCount; ++j) {
	    			truncations[j * locationCount + updatedLocation] = truncations[updatedLocation * locationCount + j];
	    		}
    		}
    	}
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
			for (int j = 0; j < locationCount; ++j) {

				const auto distance = calculateDistance<mm::MemoryManager<RealType>>(
					begin(*locationsPtr) + i * embeddingDimension,
					begin(*locationsPtr) + j * embeddingDimension,
					embeddingDimension
				);
				const auto residual = distance - observations[i * locationCount + j];
				const auto squaredResidual = residual * residual;
				squaredResiduals[i * locationCount + j] = squaredResidual;
				lSumOfSquaredResiduals += squaredResidual;

				if (withTruncation) { // compile-time check
					const auto truncation = (i == j) ? RealType(0) :
						math::logCdf<OpenCLMultiDimensionalScaling>(std::fabs(residual) * oneOverSd);
					truncations[i * locationCount + j] = truncation;
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

                return inc;
            }
		);

		sumOfSquaredResiduals += delta;
	}

// 	int count = 0
	int count2 = 0;


// 	void updateTruncations() {
//
// 		const int i = updatedLocation;
//
// 		isStoredTruncationsEmpty = false;
//
// 		auto start  = begin(*locationsPtr) + i * embeddingDimension;
// 		auto offset = begin(*locationsPtr);
//
// 		RealType delta =
// // 		accumulate_thread(0, locationCount, double(0),
//  		accumulate(0, locationCount, RealType(0),
// // 		accumulate_tbb(0, locationCount, double(0),
//
// 			[this, i, &offset, //oneOverSd,
// 			&start](const int j) {
//
//                 const auto squaredResidual = squaredResiduals[i * locationCount + j];
//
//                 const auto truncation = (i == j) ? RealType(0) :
//                 	math::logCdf<OpenCLMultiDimensionalScaling>(std::sqrt(squaredResidual) * oneOverSd);
//
//                 const auto oldTruncation = truncations[i * locationCount + j];
//                 storedTruncations[j] = oldTruncation;
//
//                 const auto inc = truncation - oldTruncation;
//
//                 truncations[i * locationCount + j] = truncation;
//                 truncations[j * locationCount + i] = truncation;
//
//                 return inc;
//             }
// 		);
//
//  		sumOfTruncations += delta;
// 	}

	void updateSumOfSquaredResidualsAndTruncations() {

		const int i = updatedLocation;
		isStoredSquaredResidualsEmpty = false;
		isStoredTruncationsEmpty = false;

		auto start  = begin(*locationsPtr) + i * embeddingDimension;
		auto offset = begin(*locationsPtr);

		std::complex<RealType> delta =

 		accumulate(0, locationCount, std::complex<RealType>(RealType(0), RealType(0)),

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

                const auto truncation = (i == j) ? RealType(0) :
                	math::logCdf<OpenCLMultiDimensionalScaling>(std::fabs(residual) * oneOverSd);

                const auto oldTruncation = truncations[i * locationCount + j];
                storedTruncations[j] = oldTruncation;

                const auto inc2 = truncation - oldTruncation;

                truncations[i * locationCount + j] = truncation;

                return std::complex<RealType>(inc, inc2);
            }
		);

		sumOfSquaredResiduals += delta.real();
 		sumOfTruncations += delta.imag();
	}

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

	template <typename Integer, typename Function, typename Real>
	inline Real accumulate(Integer begin, Integer end, Real sum, Function function) {
		for (; begin != end; ++begin) {
			sum += function(begin);
		}
		return sum;
	}

private:
	double precision;
	double storedPrecision;

	double oneOverSd;
	double storedOneOverSd;

    double sumOfSquaredResiduals;
    double storedSumOfSquaredResiduals;

    double sumOfTruncations;
    double storedSumOfTruncations;

    boost::compute::device device;
    boost::compute::context ctx;
    boost::compute::command_queue queue;

    mm::MemoryManager<RealType> observations;

    mm::MemoryManager<RealType> locations0;
    mm::MemoryManager<RealType> locations1;

    mm::MemoryManager<RealType>* locationsPtr;
    mm::MemoryManager<RealType>* storedLocationsPtr;

    mm::MemoryManager<RealType> squaredResiduals;
    mm::MemoryManager<RealType> storedSquaredResiduals;

    mm::MemoryManager<RealType> truncations;
    mm::MemoryManager<RealType> storedTruncations;

    mm::GPUMemoryManager<RealType> dObservations;

    mm::GPUMemoryManager<RealType> dLocations0;
    mm::GPUMemoryManager<RealType> dLocations1;

    mm::GPUMemoryManager<RealType>* dLocationsPtr;
    mm::GPUMemoryManager<RealType>* dStoredLocationsPtr;

    mm::GPUMemoryManager<RealType> dSquaredResiduals;
    mm::GPUMemoryManager<RealType> dStoredSquaredResiduals;

    mm::GPUMemoryManager<RealType> dTruncations;
    mm::GPUMemoryManager<RealType> dStoredTruncations;

    bool isStoredSquaredResidualsEmpty;
    bool isStoredTruncationsEmpty;

//     bool isStoredAllTruncationsEmpty;

//     int nThreads;
//     ThreadPool pool;

};

} // namespace mds

#endif // _OPENCLMULTIDIMENSIONALSCALING_HPP
