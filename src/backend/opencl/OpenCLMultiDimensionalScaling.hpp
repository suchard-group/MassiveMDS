#ifndef _OPENCLMULTIDIMENSIONALSCALING_HPP
#define _OPENCLMULTIDIMENSIONALSCALING_HPP

#include <iostream>

#include "AbstractMultiDimensionalScaling.hpp"

#include <boost/compute/algorithm/reduce.hpp>
#include "reduce_fast.hpp"

#define SSE
//#undef SSE

#define USE_VECTORS

// #define DOUBLE_CHECK

#define TILE_DIM 16

#include "OpenCLMemoryManagement.hpp"

namespace mds {

template <typename OpenCLRealType>
class OpenCLMultiDimensionalScaling : public AbstractMultiDimensionalScaling {
public:

	typedef typename OpenCLRealType::BaseType RealType;
	typedef typename OpenCLRealType::VectorType VectorType;

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
		std::cerr << "ctor OpenCLMultiDimensionalScaling" << std::endl;

		std::cerr << "All devices:" << std::endl;
		for(const auto &device : boost::compute::system::devices()){
		    std::cerr << "\t" << device.name() << std::endl;
		}

		device = boost::compute::system::default_device();
		std::cerr << "Using: " << device.name() << std::endl;

		ctx = boost::compute::context{device};
		queue = boost::compute::command_queue{ctx, device
		    , boost::compute::command_queue::enable_profiling
		    };

		dObservations = mm::GPUMemoryManager<RealType>(observations.size(), ctx);

		if (embeddingDimension != 2) {
			std::cerr << "Currently only implemented for MDS dimension == 2" << std::endl;
			exit(-1);
		}

#ifdef USE_VECTORS
		dLocations0 = mm::GPUMemoryManager<VectorType>(locations0.size() / 2, ctx);
		dLocations1 = mm::GPUMemoryManager<VectorType>(locations1.size() / 2, ctx);
#else
		dLocations0 = mm::GPUMemoryManager<RealType>(locations0.size(), ctx);
		dLocations1 = mm::GPUMemoryManager<RealType>(locations1.size(), ctx);
#endif // USE_VECTORS

		dLocationsPtr = &dLocations0;
		dStoredLocationsPtr = &dLocations1;

		dSquaredResiduals = mm::GPUMemoryManager<RealType>(squaredResiduals.size(), ctx);
		dStoredSquaredResiduals = mm::GPUMemoryManager<RealType>(storedSquaredResiduals.size(), ctx);

    	if (flags & Flags::LEFT_TRUNCATION) {
    		isLeftTruncated = true;
    		std::cout << "Using left truncation" << std::endl;

    		truncations.resize(locationCount * locationCount);
    		storedTruncations.resize(locationCount);

    		dTruncations = mm::GPUMemoryManager<RealType>(truncations.size(), ctx);
    		dStoredTruncations = mm::GPUMemoryManager<RealType>(storedTruncations.size(), ctx);
    	}
    	
		createOpenCLKernels();    	
    }

    virtual ~OpenCLMultiDimensionalScaling() {
    	std::cout << "timer1 = " << timer1 << std::endl;
    	std::cout << "timer2 = " << timer2 << std::endl;
    	std::cout << "timer3 = " << timer3 << std::endl;
    }

    void updateLocations(int locationIndex, double* location, size_t length) {

		size_t offset{0};
		size_t deviceOffset{0};

		if (locationIndex == -1) {
			// Update all locations
			assert(length == embeddingDimension * locationCount);

			residualsAndTruncationsKnown = false;
			isStoredSquaredResidualsEmpty = true;
			isStoredTruncationsEmpty = true;

			// TODO Do anything with updatedLocation?
		} else {
			// Update a single location
    		assert(length == embeddingDimension);

	    	if (updatedLocation != - 1) {
    			// more than one location updated -- do a full recomputation
	    		residualsAndTruncationsKnown = false;
	    		isStoredSquaredResidualsEmpty = true;
	    		isStoredTruncationsEmpty = true;
    		}

	    	updatedLocation = locationIndex;

			offset = locationIndex * embeddingDimension;
#ifdef USE_VECTORS
	    	deviceOffset = locationIndex;
#else
	    	deviceOffset = locationIndex * embeddingDimension;
#endif
	    }

		mm::bufferedCopy(location, location + length,
			begin(*locationsPtr) + offset,
			buffer
		);

    	// COMPUTE
    	mm::bufferedCopyToDevice(location, location + length,
    		dLocationsPtr->begin() + deviceOffset,
    		buffer, queue
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

    	// COMPUTE
    	boost::compute::copy(dLocationsPtr->begin(), dLocationsPtr->end(),
    		dStoredLocationsPtr->begin(), queue);

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

    		// COMPUTE TODO

    		if (isLeftTruncated) {
                for (int j = 0; j < locationCount; ++j) {
	    			truncations[j * locationCount + updatedLocation] = truncations[updatedLocation * locationCount + j];
	    		}

	    		// COMPUTE TODO
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

    		// COMPUTE
    		boost::compute::copy(
    			dStoredSquaredResiduals.begin(),
    			dStoredSquaredResiduals.end(),
    			dSquaredResiduals.begin() + updatedLocation * locationCount, queue
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

	    		// COMPUTE
	    		boost::compute::copy(
	    			dStoredTruncations.begin(),
	    			dStoredTruncations.end(),
	    			dTruncations.begin() + updatedLocation * locationCount, queue
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
		mm::bufferedCopy(data, data + length, begin(observations), buffer);

		// COMPUTE
		mm::bufferedCopyToDevice(data, data + length, dObservations.begin(),
			buffer, queue);

#ifdef DOUBLE_CHECK
		RealType sum = 0.0;
		boost::compute::reduce(dObservations.begin(), dObservations.end(), &sum, queue);
		RealType sum2 = std::accumulate(begin(observations), end(observations), RealType(0.0));
		std::cerr << sum << " ?= " << sum2 << std::endl;
#endif

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
// std::cerr << "A" << std::endl;
// 		RealType t1 = std::accumulate(begin(*locationsPtr), end(*locationsPtr), RealType(0));
// 		VectorType t2;
// 		boost::compute::reduce(dLocationsPtr->begin(), dLocationsPtr->end(), &t2, queue);
//
// 		std::cerr << (*locationsPtr)[0] << ", " << (*locationsPtr)[1] << std::endl;
// 		std::cerr << (*dLocationsPtr)[0] << std::endl << std::endl;
//
// 		std::cerr << (*locationsPtr)[2] << ", " << (*locationsPtr)[3] << std::endl;
// 		std::cerr << (*dLocationsPtr)[1] << std::endl << std::endl;
//
// 		std::cerr << (*locationsPtr)[2 * 6000 - 2] << ", " << (*locationsPtr)[2 * 6000 - 1] << std::endl;
// 		std::cerr << (*dLocationsPtr)[6000 - 1] << std::endl << std::endl;
//
//
// 		std::cerr << std::distance(begin(*locationsPtr), end(*locationsPtr)) << " "
// 				  << std::distance(dLocationsPtr->begin(), dLocationsPtr->end()) << std::endl;
// 		std::cerr << t1 << " " << t2 << std::endl;
//
// 		RealType t3 = 0;  RealType t5 = 0;
// 		RealType t4 = 0;  RealType t6 = 0;
// 		for (int i = 0; i < 6000; ++i) {
// 			t3 += (*locationsPtr)[2 * i];
// 			t4 += (*locationsPtr)[2 * i + 1];
//
// 			VectorType tmp = (*dLocationsPtr)[i];
// 			t5 += tmp[0];
// 			t6 += tmp[1];
// 		}
// 		std::cerr << t3 << " + " << t4 << std::endl;
// 		std::cerr << t5 << " + " << t6 << std::endl;
//
// 		exit(-1);

		RealType lSumOfSquaredResiduals = 0.0;
		RealType lSumOfTruncations = 0.0;

#ifdef DOUBLE_CHECK
		auto startTime1 = std::chrono::steady_clock::now();

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

		auto duration1 = std::chrono::steady_clock::now() - startTime1;
		if (count > 1) timer1 += std::chrono::duration<double, std::milli>(duration1).count();
#endif // DOUBLE_CHECK

//		std::cerr << "HERE2" << std::endl;
		//exit(-1);

		// COMPUTE TODO
		auto startTime2 = std::chrono::steady_clock::now();

// 		std::cerr << "Prepare for launch..." << std::endl;
#ifdef USE_VECTORS
		kernelSumOfSquaredResidualsVector.set_arg(0, *dLocationsPtr);

		if (isLeftTruncated) {
			kernelSumOfSquaredResidualsVector.set_arg(4, static_cast<RealType>(oneOverSd));
		}

		const size_t local_work_size[2] = {TILE_DIM, TILE_DIM};
		size_t work_groups = locationCount / TILE_DIM;
		if (locationCount % TILE_DIM != 0) {
			++work_groups;
		}
		const size_t global_work_size[2] = {work_groups * TILE_DIM, work_groups * TILE_DIM};

	//	std::cerr << "HERE3" << std::endl;
		//exit(-1);

		//queue.enqueue_1d_range_kernel(kernelSumOfSquaredResidualsVector, 0, locationCount * locationCount, 0);
		queue.enqueue_nd_range_kernel(kernelSumOfSquaredResidualsVector, 2, 0, global_work_size, local_work_size);
		//std::cerr << "HERE4" << std::endl;
		//exit(-1);

#else
		kernelSumOfSquaredResiduals.set_arg(0, *dLocationsPtr);
		queue.enqueue_1d_range_kernel(kernelSumOfSquaredResiduals, 0, locationCount * locationCount, 0);
#endif // USE_VECTORS

		//std::cerr << "HERE4" << std::endl;
		queue.finish();
		//std::cerr << "HERE5" << std::endl;
		auto duration2 = std::chrono::steady_clock::now() - startTime2;
		if (count > 1) timer2 += std::chrono::duration<double, std::milli>(duration2).count();


        auto startTime3 = std::chrono::steady_clock::now();
// 		std::cerr << "Done with transform." << std::endl;
		RealType sum = RealType(0.0);
		//boost::compute::reduce_fast(dSquaredResiduals.begin(), dSquaredResiduals.end(), &sum, queue);
		boost::compute::reduce(dSquaredResiduals.begin(), dSquaredResiduals.end(), &sum, queue);
//		std::cerr << "HERE6" << std::endl;

		if (isLeftTruncated) {
			RealType sum2 = RealType(0.0);
			boost::compute::reduce(dTruncations.begin(), dTruncations.end(), &sum2, queue);
			lSumOfTruncations = sum2;
		}

		queue.finish();
		auto duration3 = std::chrono::steady_clock::now() - startTime3;
		if (count > 1) timer3 += std::chrono::duration<double, std::milli>(duration3).count();


//		RealType tmp = std::accumulate(begin(squaredResiduals), end(squaredResiduals), RealType(0.0));


#ifdef DOUBLE_CHECK
  		std::cerr << sum << " - " << lSumOfSquaredResiduals << " = " <<  (sum - lSumOfSquaredResiduals) << std::endl;
#endif

//  		using namespace boost::compute;
//         boost::shared_ptr<program_cache> cache = program_cache::get_global_cache(ctx);
//
//         auto list = cache->get_keys();
//         for (auto x : list) {
//             std::cerr << x.first << " " << x.second << std::endl;
//             std::cerr << cache->get(x.first, x.second)->source() << std::endl;
//         }
//         exit(-1);


//  		std::cerr << tmp << std::endl << std::endl;
//
// 		auto d = calculateDistance<mm::MemoryManager<RealType>>(
// 					begin(*locationsPtr) + 0 * embeddingDimension,
// 					begin(*locationsPtr) + 10 * embeddingDimension,
// 					embeddingDimension
// 				);
//
// 		std::cerr << d << std::endl;
//
// 		std::cerr << dSquaredResiduals[10] << std::endl;
//
// 		std::cerr << squaredResiduals[0 * locationCount + 10] << std::endl << std::endl;
//
// 		std::cerr << (dSquaredResiduals[10] - d) << std::endl << std::endl;
//
//
// 		std::cerr << dSquaredResiduals[locationCount * locationCount - 2] << std::endl;
// 		std::cerr << squaredResiduals[locationCount * locationCount - 2] << std::endl << std::endl;

// 		for (int i = 0; i < locationCount * locationCount; ++i) {
// 			if (squaredResiduals[i] != dSquaredResiduals[i]) {
// 				std::cerr << i << " " << (squaredResiduals[i] - dSquaredResiduals[i]) << std::endl;
// 			}
// 			if (i == 100) exit(-1);
// 		}
//
// 		exit(-1);
// 		std::cerr << "Sum = " << sum << std::endl;

	    lSumOfSquaredResiduals = sum;
	    // lSumOfTruncations = // TODO

		//

    	lSumOfSquaredResiduals /= 2.0;
    	sumOfSquaredResiduals = lSumOfSquaredResiduals;

    	if (withTruncation) {
    		lSumOfTruncations /= 2.0;
    		sumOfTruncations = lSumOfTruncations;
    	}

	    residualsAndTruncationsKnown = true;
	    sumsOfResidualsAndTruncationsKnown = true;

	    count++;
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

		// COMPUTE TODO

		sumOfSquaredResiduals += delta;
	}

// 	int count = 0
	int count2 = 0;

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

		// COMPUTE TODO

		sumOfSquaredResiduals += delta.real();
 		sumOfTruncations += delta.imag();
	}

#ifdef SSE
    template <typename HostVectorType, typename Iterator>
    double calculateDistance(Iterator iX, Iterator iY, int length) const {

        using AlignedValueType = typename HostVectorType::allocator_type::aligned_value_type;

        auto sum = static_cast<AlignedValueType>(0);
        AlignedValueType* x = &*iX;
        AlignedValueType* y = &*iY;

        for (int i = 0; i < 2; ++i, ++x, ++y) {
            const auto difference = *x - *y; // TODO Why does this seg-fault?
            sum += difference * difference;
        }
        return std::sqrt(sum);
    }
#else // SSE
    template <typename HostVectorType, typename Iterator>
    double calculateDistance(Iterator x, Iterator y, int length) const {
        auto sum = static_cast<RealType>(0);

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


	void createOpenCLKernels() {

		const char Test[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
			// approximation of the cumulative normal distribution function
			static float cnd(float d)
			{
				const float A1 =  0.319381530f;
				const float A2 = -0.356563782f;
				const float A3 =  1.781477937f;
				const float A4 = -1.821255978f;
				const float A5 =  1.330274429f;
				const float RSQRT2PI = 0.39894228040143267793994605993438f;

				float K = 1.0f / (1.0f + 0.2316419f * fabs(d));
				float cnd =
					RSQRT2PI * exp(-0.5f * d * d) *
					(K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

				if(d > 0){
					cnd = 1.0f - cnd;
				}

				return cnd;
			}
		);

		const char cdfString1Double[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
			static double cdf(double);

			static double cdf(double value) {
	    		return 0.5 * erfc(-value * M_SQRT1_2); 
	    	}
		);

		const char cdfString1Float[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
			static float cdf(float);

			static float cdf(float value) {	   
			
				const float rsqrt2f =  0.70710678118655f;	
	    		return 0.5f * erfc(-value * rsqrt2f);
	    	}
		);

		const char SumOfSquaredResidualsKernelVectorBody[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
				const uint offsetJ = get_group_id(0) * TILE_DIM;
				const uint offsetI = get_group_id(1) * TILE_DIM;

				const uint j = offsetJ + get_local_id(0);
				const uint i = offsetI + get_local_id(1);

				__local REAL_VECTOR tile[2][TILE_DIM + 1]; // tile[0] == locations_j, tile[1] == locations_i

				if (get_local_id(1) < 2) { // load just 2 rows
					tile[get_local_id(1)][get_local_id(0)] = locations[
						(get_local_id(1) - 0) * (offsetI + get_local_id(0)) + // tile[1] = locations_i
						(1 - get_local_id(1)) * (offsetJ + get_local_id(0))   // tile[0] = locations_j
					];
				}

				barrier(CLK_LOCAL_MEM_FENCE);

				if (i < locationCount && j < locationCount) {

					const REAL distance = length(
						tile[1][get_local_id(1)] - tile[0][get_local_id(0)]
// 						locations[i] - locations[j]
					);

					const REAL residual = distance - observations[i * locationCount + j];
					const REAL squaredResidual = residual * residual;

					squaredResiduals[i * locationCount + j] = squaredResidual;
				}
			}
		);

		bool useLocalMemory = false;

// 		std::cerr << "A" << std::endl;

		std::stringstream code;
		std::stringstream options;

		options << "-DTILE_DIM=" << TILE_DIM;
		if (useLocalMemory) {
	    	options << " -DLOCAL_MEM";
	    }

		if (sizeof(RealType) == 8) { // 64-bit fp
			code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
			options << " -DREAL=double -DREAL_VECTOR=double2 -DZERO=0.0";
			
			code << cdfString1Double;
			
		} else { // 32-bit fp
			options << " -DREAL=float -DREAL_VECTOR=float2 -DZERO=0.0f";
			
			code << cdfString1Float;
		}
				
		code << 
			" __kernel void computeSSR(__global const REAL_VECTOR *locations,  \n" <<
			"  						   __global const REAL *observations,      \n" <<
			"						   __global REAL *squaredResiduals,        \n"; 
		
		if (isLeftTruncated) {
			code << 
			"						   __global REAL *truncations,             \n"	<<
			"                          const REAL oneOverSd,                   \n";				
		}
		code << 
			"						   const uint locationCount) {            \n";
									
		code << BOOST_COMPUTE_STRINGIZE_SOURCE(
				const uint offsetJ = get_group_id(0) * TILE_DIM;
				const uint offsetI = get_group_id(1) * TILE_DIM;

				const uint j = offsetJ + get_local_id(0);
				const uint i = offsetI + get_local_id(1);

				__local REAL_VECTOR tile[2][TILE_DIM + 1]; // tile[0] == locations_j, tile[1] == locations_i

				if (get_local_id(1) < 2) { // load just 2 rows
					tile[get_local_id(1)][get_local_id(0)] = locations[
						(get_local_id(1) - 0) * (offsetI + get_local_id(0)) + // tile[1] = locations_i
						(1 - get_local_id(1)) * (offsetJ + get_local_id(0))   // tile[0] = locations_j
					];
				}

				barrier(CLK_LOCAL_MEM_FENCE);

				if (i < locationCount && j < locationCount) {

					const REAL distance = length(
						tile[1][get_local_id(1)] - tile[0][get_local_id(0)]
// 						locations[i] - locations[j]
					);

					const REAL residual = distance - observations[i * locationCount + j];
					const REAL squaredResidual = residual * residual;

					squaredResiduals[i * locationCount + j] = squaredResidual;
		);
		
		if (isLeftTruncated) {
			code << BOOST_COMPUTE_STRINGIZE_SOURCE(
					const REAL truncation = (i == j) ? ZERO : log(cdf(fabs(residual) * oneOverSd));;
					truncations[i * locationCount + j] = truncation;			
			);
		}
		
		code << BOOST_COMPUTE_STRINGIZE_SOURCE(
				}
			}
		);				

		program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
	    	kernelSumOfSquaredResidualsVector = boost::compute::kernel(program, "computeSSR");

#ifdef DOUBLE_CHECK
		std::cerr << kernelSumOfSquaredResidualsVector.get_program().source() << std::endl;
		//exit(-1);
#endif // DOUBLE_CHECK

		int index = 0;
		kernelSumOfSquaredResidualsVector.set_arg(index++, dLocations0); // TODO Must update
		kernelSumOfSquaredResidualsVector.set_arg(index++, dObservations);
		kernelSumOfSquaredResidualsVector.set_arg(index++, dSquaredResiduals);
		if (isLeftTruncated) {
			kernelSumOfSquaredResidualsVector.set_arg(index++, dTruncations);
			kernelSumOfSquaredResidualsVector.set_arg(index++, static_cast<RealType>(oneOverSd));
		}
		kernelSumOfSquaredResidualsVector.set_arg(index++, boost::compute::uint_(locationCount));


#ifdef DOUBLE_CHECK
 		using namespace boost::compute;
        boost::shared_ptr<program_cache> cache = program_cache::get_global_cache(ctx);

		RealType sum = RealType(0.0);
		boost::compute::reduce(dSquaredResiduals.begin(), dSquaredResiduals.end(), &sum, queue);

		auto programInfo = *begin(cache->get_keys());
		std::cerr << "Try " << programInfo.first << " : " << programInfo.second << std::endl;

        boost::compute::program programReduce = *cache->get(programInfo.first, programInfo.second);
        auto kernelReduce = kernel(programReduce, "reduce");
        std::cerr << programReduce.source() << std::endl;


        const auto &device2 = queue.get_device();
        std::cerr << "nvidia? " << detail::is_nvidia_device(device) << " " << device.name() << " " << device.vendor() << std::endl;
        std::cerr << "nvidia? " << detail::is_nvidia_device(device2) << " " << device2.name() << " " << device.vendor() << std::endl;

		std::cerr << "Done compile VECTOR." << std::endl;
#endif

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

#ifdef USE_VECTORS
    mm::GPUMemoryManager<VectorType> dLocations0;
    mm::GPUMemoryManager<VectorType> dLocations1;

    mm::GPUMemoryManager<VectorType>* dLocationsPtr;
    mm::GPUMemoryManager<VectorType>* dStoredLocationsPtr;
#else
    mm::GPUMemoryManager<RealType> dLocations0;
    mm::GPUMemoryManager<RealType> dLocations1;

    mm::GPUMemoryManager<RealType>* dLocationsPtr;
    mm::GPUMemoryManager<RealType>* dStoredLocationsPtr;
#endif // USE_VECTORS


    mm::GPUMemoryManager<RealType> dSquaredResiduals;
    mm::GPUMemoryManager<RealType> dStoredSquaredResiduals;

    mm::GPUMemoryManager<RealType> dTruncations;
    mm::GPUMemoryManager<RealType> dStoredTruncations;

    bool isStoredSquaredResidualsEmpty;
    bool isStoredTruncationsEmpty;

    mm::MemoryManager<RealType> buffer;

    boost::compute::program program;

#ifdef USE_VECTORS
	boost::compute::kernel kernelSumOfSquaredResidualsVector;
#else
    boost::compute::kernel kernelSumOfSquaredResiduals;
#endif // USE_VECTORS

	double timer1 = 0;
	double timer2 = 0;
	double timer3 = 0;


//     bool isStoredAllTruncationsEmpty;

//     int nThreads;
//     ThreadPool pool;


};

} // namespace mds

#endif // _OPENCLMULTIDIMENSIONALSCALING_HPP
