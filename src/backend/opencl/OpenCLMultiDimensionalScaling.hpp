#ifndef _OPENCL_MULTIDIMENSIONAL_SCALING_HPP
#define _OPENCL_MULTIDIMENSIONAL_SCALING_HPP

#include <iostream>
#include <cmath>

#include "AbstractMultiDimensionalScaling.hpp"

#include <boost/compute/algorithm/reduce.hpp>
#include "reduce_fast.hpp"

#ifdef RBUILD
#include <Rcpp.h>
#endif

#define SSE
//#undef SSE

#define USE_VECTORS

//#define DOUBLE_CHECK

//#define DOUBLE_CHECK_GRADIENT

#define TILE_DIM 16

#define TILE_DIM_I  128
//#define TILE_DIM_J  128
#define TPB 128
#define DELTA 1;

#define USE_VECTOR

//#define DEBUG_KERNELS

#include "OpenCLMemoryManagement.hpp"
#include "Reducer.hpp"

#include <boost/compute/algorithm/accumulate.hpp>

namespace mds {

template <typename OpenCLRealType>
class OpenCLMultiDimensionalScaling : public AbstractMultiDimensionalScaling {
public:

	typedef typename OpenCLRealType::BaseType RealType;
	typedef typename OpenCLRealType::VectorType VectorType;

    OpenCLMultiDimensionalScaling(int embeddingDimension, int locationCount, long flags, int deviceNumber)
        : AbstractMultiDimensionalScaling(embeddingDimension, locationCount, flags, bandwidth),
          precision(0.0), storedPrecision(0.0),
          oneOverSd(0.0), storedOneOverSd(0.0),
          sumOfSquaredResiduals(0.0), storedSumOfSquaredResiduals(0.0),
          sumOfTruncations(0.0), storedSumOfTruncations(0.0),

          observations(locationCount * locationCount),

          locations0(locationCount * OpenCLRealType::dim),
		  locations1(locationCount * OpenCLRealType::dim),
		  locationsPtr(&locations0),
		  storedLocationsPtr(&locations1),

          squaredResiduals(locationCount * locationCount),
          storedSquaredResiduals(locationCount),

          isStoredSquaredResidualsEmpty(false),
          isStoredTruncationsEmpty(false)//,
//          isStoredAllTruncationsEmpty(false)

//           , nThreads(4) //, pool(nThreads)
    {
#ifdef RBUILD
      Rcpp::Rcout << "ctor OpenCLMultiDimensionalScaling" << std::endl;

      Rcpp::Rcout << "All devices:" << std::endl;

      const auto devices = boost::compute::system::devices();

      for(const auto &device : devices){
        Rcpp::Rcout << "\t" << device.name() << std::endl;
      }


      if (deviceNumber < 0 || deviceNumber >= devices.size()) {
        device = boost::compute::system::default_device();
      } else {
        device = devices[deviceNumber];
      }

      //device = devices[devices.size() - 1]; // hackishly chooses correct device TODO do this correctly

      if (device.type()!=CL_DEVICE_TYPE_GPU){
          Rcpp::stop("Error: selected device not GPU.");
      } else {
          Rcpp::Rcout << "Using: " << device.name() << std::endl;
      }

      ctx = boost::compute::context(device, 0);
      queue = boost::compute::command_queue{ctx, device
        , boost::compute::command_queue::enable_profiling
      };

      dObservations = mm::GPUMemoryManager<RealType>(observations.size(), ctx);

      Rcpp::Rcout << "\twith vector-dim = " << OpenCLRealType::dim << std::endl;

#else //RBUILD
      std::cerr << "ctor OpenCLMultiDimensionalScaling" << std::endl;

      std::cerr << "All devices:" << std::endl;

      const auto devices = boost::compute::system::devices();

      for(const auto &device : devices){
        std::cerr << "\t" << device.name() << std::endl;
      }


      if (deviceNumber < 0 || deviceNumber >= devices.size()) {
        device = boost::compute::system::default_device();
      } else {
        device = devices[deviceNumber];
      }

      //device = devices[devices.size() - 1]; // hackishly chooses correct device TODO do this correctly

      if (device.type()!=CL_DEVICE_TYPE_GPU){
          std::cerr << "Error: selected device not GPU." << std::endl;
          exit(-1);
      } else {
          std::cerr << "Using: " << device.name() << std::endl;
      }

      ctx = boost::compute::context(device, 0);
      queue = boost::compute::command_queue{ctx, device
        , boost::compute::command_queue::enable_profiling
      };

      dObservations = mm::GPUMemoryManager<RealType>(observations.size(), ctx);

      std::cerr << "\twith vector-dim = " << OpenCLRealType::dim << std::endl;
#endif //RBUILD

//		if (embeddingDimension != OpenCLRealType::dim) {
//			std::cerr << "Currently only implemented for embedding dimension == VectorDimension" << std::endl;
//			exit(-1);
//		}

#ifdef USE_VECTORS
		dLocations0 = mm::GPUMemoryManager<VectorType>(locationCount, ctx);
		dLocations1 = mm::GPUMemoryManager<VectorType>(locationCount, ctx);
		dGradient   = mm::GPUMemoryManager<VectorType>(locationCount, ctx);
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
#ifdef RBUILD
    	  Rcpp::Rcout << "Using left truncation" << std::endl;
#else
    		std::cout << "Using left truncation" << std::endl;
#endif

    		truncations.resize(locationCount * locationCount);
    		storedTruncations.resize(locationCount);

    		dTruncations = mm::GPUMemoryManager<RealType>(truncations.size(), ctx);
    		dStoredTruncations = mm::GPUMemoryManager<RealType>(storedTruncations.size(), ctx);
    	}

		createOpenCLKernels();
    }

//    virtual ~OpenCLMultiDimensionalScaling() override {
//#ifdef RBUILD
//      Rcpp::Rcout << "timer1 = " << timer1 << std::endl;
//      Rcpp::Rcout << "timer2 = " << timer2 << std::endl;
//      Rcpp::Rcout << "timer3 = " << timer3 << std::endl;
//#else
//    	std::cout << "timer1 = " << timer1 << std::endl;
//    	std::cout << "timer2 = " << timer2 << std::endl;
//    	std::cout << "timer3 = " << timer3 << std::endl;
//#endif
//    }

    void updateLocations(int locationIndex, double* location, size_t length) override {

		size_t offset{0};
		size_t deviceOffset{0};

		if (locationIndex == -1) {
			// Update all locations
			assert(length == OpenCLRealType::dim * locationCount ||
                    length == embeddingDimension * locationCount);

			incrementsKnown = false;
			isStoredSquaredResidualsEmpty = true;
			isStoredTruncationsEmpty = true;

		} else {
			// Update a single location
    		assert(length == OpenCLRealType::dim ||
                    length == embeddingDimension);

	    	if (updatedLocation != - 1) {
    			// more than one location updated -- do a full recomputation
	    		incrementsKnown = false;
	    		isStoredSquaredResidualsEmpty = true;
	    		isStoredTruncationsEmpty = true;
    		}

	    	updatedLocation = locationIndex;

			offset = locationIndex * OpenCLRealType::dim;
#ifdef USE_VECTORS
	    	deviceOffset = static_cast<size_t>(locationIndex);
#else
	    	deviceOffset = locationIndex * embeddingDimension;
#endif
	    }

        // If requires padding
        if (embeddingDimension != OpenCLRealType::dim) {
            if (locationIndex == -1) {

                mm::paddedBufferedCopy(location, embeddingDimension, embeddingDimension,
                                       begin(*locationsPtr) + offset, OpenCLRealType::dim,
                                       locationCount, buffer);

                length = OpenCLRealType::dim * locationCount; // New padded length

            } else {

                mm::bufferedCopy(location, location + length,
                                 begin(*locationsPtr) + offset,
                                 buffer);

                length = OpenCLRealType::dim;
            }
        } else {
            // Without padding
            mm::bufferedCopy(location, location + length,
                             begin(*locationsPtr) + offset,
                             buffer
            );
        }

    	// COMPUTE
        mm::copyToDevice<OpenCLRealType>(begin(*locationsPtr) + offset,
                                         begin(*locationsPtr) + offset + length,
                                         dLocationsPtr->begin() + deviceOffset,
                                         queue
        );

    	sumOfIncrementsKnown = false;
    }

    int getInternalDimension() override { return OpenCLRealType::dim; }

	void getLogLikelihoodGradient(double* result, size_t length) override {

        // TODO Buffer gradients

#ifdef DOUBLE_CHECK_GRADIENT
		assert(length == locationCount * embeddingDimension ||
                       length == locationCount * OpenCLRealType::dim);

		if (gradient.size() != locationCount * OpenCLRealType::dim) {
			gradient.resize(locationCount * OpenCLRealType::dim);
		}

		std::fill(std::begin(gradient), std::end(gradient), static_cast<RealType>(0.0));

		const RealType scale = precision;

		for (int i = 0; i < locationCount; ++i) {
			for (int j = 0; j < locationCount; ++j) {
				if (i != j) {
					const auto distance = calculateDistance<mm::MemoryManager<RealType>>(
							begin(*locationsPtr) + i * OpenCLRealType::dim,
							begin(*locationsPtr) + j * OpenCLRealType::dim
					);

					const RealType dataContribution =
							(observations[i * locationCount + j] - distance) * scale / distance;

                    for (int d = 0; d < embeddingDimension; ++d) {
                        const RealType update = dataContribution *
                                                 ((*locationsPtr)[i * OpenCLRealType::dim + d] - (*locationsPtr)[j * OpenCLRealType::dim + d]);
                        gradient[i * OpenCLRealType::dim + d] += update;
                    }
				}
			}
		}

		if (doubleBuffer.size() != locationCount * OpenCLRealType::dim) {
			doubleBuffer.resize(locationCount * OpenCLRealType::dim);
		}

		mm::bufferedCopy(std::begin(gradient), std::end(gradient), doubleBuffer.data(), buffer);

        std::vector<double> testGradient0;
        for (int d = 0; d < embeddingDimension; ++d) {
            testGradient0.push_back(doubleBuffer[d]);
        }
        for (int d = 0; d < embeddingDimension; ++d) {
            testGradient0.push_back(doubleBuffer[OpenCLRealType::dim * (locationCount - 1) + d]);
        }

        std::vector<double> testGradient00;
        for (int d = 0; d < OpenCLRealType::dim; ++d) {
            testGradient00.push_back(doubleBuffer[d]);
        }
        for (int d = 0; d < OpenCLRealType::dim; ++d) {
            testGradient00.push_back(doubleBuffer[OpenCLRealType::dim * (locationCount - 1) + d]);
        }

#endif // DOUBLE_CHECK_GRADIENT

		kernelGradientVector.set_arg(0, *dLocationsPtr);
		kernelGradientVector.set_arg(3, static_cast<RealType>(precision));

		queue.enqueue_1d_range_kernel(kernelGradientVector, 0,
                                      static_cast<unsigned int>(locationCount) * TPB, TPB);
		queue.finish();

        if (length == locationCount * OpenCLRealType::dim) {

            mm::bufferedCopyFromDevice<OpenCLRealType>(dGradient.begin(), dGradient.end(),
                                       result, buffer, queue);
            queue.finish();

        } else {

            if (doubleBuffer.size() != locationCount * OpenCLRealType::dim) {
                doubleBuffer.resize(locationCount * OpenCLRealType::dim);
            }

            mm::bufferedCopyFromDevice<OpenCLRealType>(dGradient.begin(), dGradient.end(),
                                       doubleBuffer.data(), buffer, queue);
            queue.finish();

            mm::paddedBufferedCopy(begin(doubleBuffer), OpenCLRealType::dim, embeddingDimension,
                                   result, embeddingDimension,
                                   locationCount, buffer);
        }

#ifdef DOUBLE_CHECK_GRADIENT
        std::vector<double> testGradient1;
        for (int i = 0; i < embeddingDimension; ++i) {
            testGradient1.push_back(result[i]);
        }

		int stride = (length == locationCount * OpenCLRealType::dim) ?
					 OpenCLRealType::dim : embeddingDimension;

        for (int i = 0; i < embeddingDimension; ++i) {
            testGradient1.push_back(result[stride * (locationCount - 1) + i]);
        }

        std::vector<double> testGradient11;
        for (int i = 0; i < OpenCLRealType::dim; ++i) {
            testGradient11.push_back(doubleBuffer[i]);
        }

#ifdef RBUILD
        Rcpp::Rcout << "cpu0: ";
        for (auto x : testGradient0) {
          Rcpp::Rcout << " " << x;
        }
        Rcpp::Rcout << std::endl;

        Rcpp::Rcout << "cpu1: ";
        for (auto x : testGradient00) {
          Rcpp::Rcout << " " << x;
        }
        Rcpp::Rcout << std::endl;

        Rcpp::Rcout << "gpu0: ";
        for (auto x : testGradient1) {
          Rcpp::Rcout << " " << x;
        }
        Rcpp::Rcout << std::endl;

        Rcpp::Rcout << "gpu1: ";
        for (auto x : testGradient11) {
          Rcpp::Rcout << " " << x;
        }
        Rcpp::Rcout << std::endl;
#else //RBUILD
        std::cerr << "cpu0: ";
        for (auto x : testGradient0) {
            std::cerr << " " << x;
        }
        std::cerr << std::endl;

        std::cerr << "cpu1: ";
        for (auto x : testGradient00) {
            std::cerr << " " << x;
        }
        std::cerr << std::endl;

        std::cerr << "gpu0: ";
        for (auto x : testGradient1) {
            std::cerr << " " << x;
        }
        std::cerr << std::endl;

        std::cerr << "gpu1: ";
        for (auto x : testGradient11) {
            std::cerr << " " << x;
        }
        std::cerr << std::endl;
#endif //RBUILD
#endif

 	}

    void computeResidualsAndTruncations() {

		if (!incrementsKnown) {
			if (isLeftTruncated) { // run-time dispatch to compile-time optimization
				computeSumOfSquaredResiduals<true>();
			} else {
				computeSumOfSquaredResiduals<false>();
			}
			incrementsKnown = true;
		} else {
#ifdef RBUILD
		  Rcpp::Rcout << "SHOULD NOT BE HERE" << std::endl;
#else
			std::cerr << "SHOULD NOT BE HERE" << std::endl;
#endif
			if (isLeftTruncated) {
				updateSumOfSquaredResidualsAndTruncations();
			} else {
				updateSumOfSquaredResiduals();
			}
		}
    }

    double getSumOfIncrements() override {
    	if (!sumOfIncrementsKnown) {
			computeResidualsAndTruncations();
			sumOfIncrementsKnown = true;
		}
		if (isLeftTruncated) {
			return sumOfSquaredResiduals;
		} else {
			return 0.5 * precision * sumOfSquaredResiduals;
		}
 	}    // TODO Duplicated code with CPU version; there is a problem here?

 	double getSumOfLogTruncations() {
    	if (!sumOfIncrementsKnown) {
			computeResidualsAndTruncations();
			sumOfIncrementsKnown = true;
		}
 		return sumOfTruncations;
 	}

    void storeState() override {
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

    void acceptState() override {
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

    void restoreState() override {
    	sumOfSquaredResiduals = storedSumOfSquaredResiduals;
    	sumOfIncrementsKnown = true;

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

    		incrementsKnown = true;
    	} else {
    		incrementsKnown = false; // Force recompute;
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

    	// COMPUTE
    	auto tmp2 = dStoredLocationsPtr;
    	dStoredLocationsPtr = dLocationsPtr;
    	dLocationsPtr = tmp2;

    }

    void setPairwiseData(double* data, size_t length) override {
		assert(length == observations.size());
		mm::bufferedCopy(data, data + length, begin(observations), buffer);

		// COMPUTE
		mm::bufferedCopyToDevice(data, data + length, dObservations.begin(),
			buffer, queue);

#ifdef DOUBLE_CHECK
		RealType sum = 0.0;
		boost::compute::reduce(dObservations.begin(), dObservations.end(), &sum, queue);
		RealType sum2 = std::accumulate(begin(observations), end(observations), RealType(0.0));
#ifdef RBUILD
		Rcpp::Rcout << sum << " ?= " << sum2 << std::endl;
#else
		std::cerr << sum << " ?= " << sum2 << std::endl;
#endif
#endif

    }

    void setParameters(double* data, size_t length) override {
		assert(length == 1); // Call only with precision
		precision = data[0]; // TODO Remove
		oneOverSd = std::sqrt(data[0]);

		// Handle truncations
		if (isLeftTruncated) {
			incrementsKnown = false;
			sumOfIncrementsKnown = false;

    		isStoredSquaredResidualsEmpty = true;
    		isStoredTruncationsEmpty = true;

		}
    }

    void makeDirty() override {
    	sumOfIncrementsKnown = false;
    	incrementsKnown = false;
    }

	int count = 0;

	template <bool withTruncation>
	void computeSumOfSquaredResiduals() {

		RealType lSumOfSquaredResiduals = 0.0;

#ifdef DOUBLE_CHECK
	  RealType lSumOfTruncations = 0.0;
		auto startTime1 = std::chrono::steady_clock::now();

		for (int i = 0; i < locationCount; ++i) { // TODO Parallelize
			for (int j = 0; j < locationCount; ++j) {

				const auto distance = calculateDistance<mm::MemoryManager<RealType>>(
					begin(*locationsPtr) + i * OpenCLRealType::dim,
					begin(*locationsPtr) + j * OpenCLRealType::dim
				);
				const auto residual = !std::isnan(observations[i * locationCount + j]) *
				        (distance - observations[i * locationCount + j]);
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
			kernelSumOfSquaredResidualsVector.set_arg(3, static_cast<RealType>(precision));
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

// 		if (isLeftTruncated) {
// 			RealType sum2 = RealType(0.0);
// 			boost::compute::reduce(dTruncations.begin(), dTruncations.end(), &sum2, queue);
// 			lSumOfTruncations = sum2;
// 		}

		queue.finish();
		auto duration3 = std::chrono::steady_clock::now() - startTime3;
		if (count > 1) timer3 += std::chrono::duration<double, std::milli>(duration3).count();


//		RealType tmp = std::accumulate(begin(squaredResiduals), end(squaredResiduals), RealType(0.0));


#ifdef DOUBLE_CHECK
#ifdef RBUILD
      Rcpp::Rcout << sum << " - " << lSumOfSquaredResiduals << " = " <<  (sum - lSumOfSquaredResiduals) << std::endl;
#else
  		std::cerr << sum << " - " << lSumOfSquaredResiduals << " = " <<  (sum - lSumOfSquaredResiduals) << std::endl;
#endif
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
// 					begin(*locationsPtr) + 0 * OpenCLRealType::dim,
// 					begin(*locationsPtr) + 10 * OpenCLRealType::dim
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
	    // lSumOfTruncations =

		//

    	lSumOfSquaredResiduals /= 2.0;
    	sumOfSquaredResiduals = lSumOfSquaredResiduals;

//     	if (withTruncation) {
//     		lSumOfTruncations /= 2.0;
//     		sumOfTruncations = lSumOfTruncations;
//     	}

// 		std::cerr << lSumOfSquaredResiduals << " " << lSumOfTruncations << std::endl;
// 		exit(-1);

	    incrementsKnown = true;
	    sumOfIncrementsKnown = true;

	    count++;
	}

	void updateSumOfSquaredResiduals() {
		// double delta = 0.0;

		const int i = updatedLocation;
		isStoredSquaredResidualsEmpty = false;

		auto start  = begin(*locationsPtr) + i * OpenCLRealType::dim;
		auto offset = begin(*locationsPtr);

		RealType delta =

// 		accumulate_thread(0, locationCount, double(0),
 		accumulate(0, locationCount, RealType(0),
// 		accumulate_tbb(0, locationCount, double(0),

			[this, i, &offset,
			&start](const int j) {
                const auto distance = calculateDistance<mm::MemoryManager<RealType>>(
                    start,
                    offset + OpenCLRealType::dim * j
                );

                const auto residual = !std::isnan(observations[i * locationCount + j]) * // Andrew's not sure...
                        (distance - observations[i * locationCount + j]);
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
//	int count2 = 0;

	void updateSumOfSquaredResidualsAndTruncations() {

		const int i = updatedLocation;
		isStoredSquaredResidualsEmpty = false;
		isStoredTruncationsEmpty = false;

		auto start  = begin(*locationsPtr) + i * OpenCLRealType::dim;
		auto offset = begin(*locationsPtr);

		std::complex<RealType> delta =

 		accumulate(0, locationCount, std::complex<RealType>(RealType(0), RealType(0)),

			[this, i, &offset, //oneOverSd,
			&start](const int j) {
                const auto distance = calculateDistance<mm::MemoryManager<RealType>>(
                    start,
                    offset + OpenCLRealType::dim * j
                );

                const auto residual = !std::isnan(observations[i * locationCount + j]) * // Andrew's not sure...
                                      (distance - observations[i * locationCount + j]);
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
    double calculateDistance(Iterator iX, Iterator iY) const {

        using AlignedValueType = typename HostVectorType::allocator_type::aligned_value_type;

        auto sum = static_cast<AlignedValueType>(0);
        AlignedValueType* x = &*iX;
        AlignedValueType* y = &*iY;

        for (int i = 0; i < OpenCLRealType::dim; ++i, ++x, ++y) {
            const auto difference = *x - *y; // TODO Why does this seg-fault?
            sum += difference * difference;
        }
        return std::sqrt(sum);
    }

//    template <typename HostVectorType, typename Iterator>
//    double calculateDistance<HostVectorType, Iterator, 2>(Iterator iX, Iterator iY, int length) const {
//
//        using AlignedValueType = typename HostVectorType::allocator_type::aligned_value_type;
//
//        auto sum = static_cast<AlignedValueType>(0);
//        AlignedValueType* x = &*iX;
//        AlignedValueType* y = &*iY;
//
//        for (int i = 0; i < 2; ++i, ++x, ++y) {
//            const auto difference = *x - *y; // TODO Why does this seg-fault?
//            sum += difference * difference;
//        }
//        return std::sqrt(sum);
//    }


#else // SSE
    template <typename HostVectorType, typename Iterator>
    double calculateDistance(Iterator x, Iterator y, int length) const {

        assert (false);

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

	void createOpenCLLikelihoodKernel() {

//		const char Test[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
//			// approximation of the cumulative normal distribution function
//			static float cnd(float d)
//			{
//				const float A1 =  0.319381530f;
//				const float A2 = -0.356563782f;
//				const float A3 =  1.781477937f;
//				const float A4 = -1.821255978f;
//				const float A5 =  1.330274429f;
//				const float R_SQRT_2PI = 0.39894228040143267793994605993438f;
//
//				float K = 1.0f / (1.0f + 0.2316419f * fabs(d));
//				float cnd =
//                        R_SQRT_2PI * exp(-0.5f * d * d) *
//					(K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
//
//				if(d > 0){
//					cnd = 1.0f - cnd;
//				}
//
//				return cnd;
//			}
//		);

		const char cdfString1Double[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
			static double cdf(double);

			static double cdf(double value) {
	    		return 0.5 * erfc(-value * M_SQRT1_2);
	    	}
		);

		const char cdfString1Float[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
			static float cdf(float);

			static float cdf(float value) {

				const float rSqrt2f = 0.70710678118655f;
	    		return 0.5f * erfc(-value * rSqrt2f);
	    	}
		);

//		const char SumOfSquaredResidualsKernelVectorBody[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
//				const uint offsetJ = get_group_id(0) * TILE_DIM;
//				const uint offsetI = get_group_id(1) * TILE_DIM;
//
//				const uint j = offsetJ + get_local_id(0);
//				const uint i = offsetI + get_local_id(1);
//
//				__local REAL_VECTOR tile[2][TILE_DIM + 1]; // tile[0] == locations_j, tile[1] == locations_i
//
//				if (get_local_id(1) < 2) { // load just 2 rows
//					tile[get_local_id(1)][get_local_id(0)] = locations[
//						(get_local_id(1) - 0) * (offsetI + get_local_id(0)) + // tile[1] = locations_i
//						(1 - get_local_id(1)) * (offsetJ + get_local_id(0))   // tile[0] = locations_j
//					];
//				}
//
//				barrier(CLK_LOCAL_MEM_FENCE);
//
//				if (i < locationCount && j < locationCount) {
//
//					const REAL distance = length(
//						tile[1][get_local_id(1)] - tile[0][get_local_id(0)]
//// 						locations[i] - locations[j]
//					);
//
//					const REAL residual =  !isnan(observations[i * locationCount + j]) *
//					        (distance - observations[i * locationCount + j]); //Andrew's not sure
//					const REAL squaredResidual = residual * residual;
//
//					squaredResiduals[i * locationCount + j] = squaredResidual;
//				}
//			}
//		);

//		bool useLocalMemory = false;

// 		std::cerr << "A" << std::endl;

		std::stringstream code;
		std::stringstream options;

		options << "-DTILE_DIM=" << TILE_DIM;
//		if (useLocalMemory) {
//	    	options << " -DLOCAL_MEM";
//	    }

		if (sizeof(RealType) == 8) { // 64-bit fp
			code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
			options << " -DREAL=double -DREAL_VECTOR=double" << OpenCLRealType::dim << " -DCAST=long"
                    << " -DZERO=0.0 -DHALF=0.5";

			if (isLeftTruncated) {
				code << cdfString1Double;
			}

		} else { // 32-bit fp
			options << " -DREAL=float -DREAL_VECTOR=float" << OpenCLRealType::dim << " -DCAST=int"
                    << " -DZERO=0.0f -DHALF=0.5f";

			if (isLeftTruncated) {
				code << cdfString1Float;
			}
		}

		code <<
			" __kernel void computeSSR(__global const REAL_VECTOR *locations,  \n" <<
			"  						   __global const REAL *observations,      \n" <<
			"						   __global REAL *squaredResiduals,        \n";


		if (isLeftTruncated) {
			code <<
            "                          const REAL precision,                   \n" <<
			"                          const REAL oneOverSd,                   \n";
		}

		code <<
			"						   const uint locationCount) {            \n";


//        code << "}\n";

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

                    const REAL_VECTOR difference = tile[1][get_local_id(1)] - tile[0][get_local_id(0)];
                    // 						locations[i] - locations[j]
        );

        if (OpenCLRealType::dim == 8) {
            code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                    const REAL distance = sqrt(
                            dot(difference.lo, difference.lo) +
                            dot(difference.hi, difference.hi)
                    );
            );
        } else {
            code << BOOST_COMPUTE_STRINGIZE_SOURCE(
					const REAL distance = length(difference);
            );
        }

        code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                    const REAL observation = observations[i * locationCount + j];

                    const REAL residual = select(distance - observation, ZERO, (CAST)isnan(observation));
                    //const REAL residual = distance - observation;
                    REAL squaredResidual = residual * residual;

//                    if (!isnan(observation)) {
//                        const REAL residual = (distance - observation);
//                        squaredResidual = residual * residual;
//                    }
        );

		if (isLeftTruncated) {
			code << BOOST_COMPUTE_STRINGIZE_SOURCE(
					squaredResidual *= HALF * precision;
					const REAL truncation = (i == j) ? ZERO :
					        select(log(cdf(fabs(distance) * oneOverSd)), ZERO, (CAST)isnan(observation));
					squaredResidual += truncation;
			);
		}

		code << BOOST_COMPUTE_STRINGIZE_SOURCE(
			 		squaredResiduals[i * locationCount + j] = squaredResidual;
				}
            }
		);

#ifdef DEBUG_KERNELS
#ifdef RBUILD
		    Rcpp::Rcout << "Likelihood kernel\n" << code.str() << std::endl;
#else
        std::cerr << "Likelihood kernel\n" << code.str() << std::endl;
#endif
#endif

		program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
	    	kernelSumOfSquaredResidualsVector = boost::compute::kernel(program, "computeSSR");

#ifdef DEBUG_KERNELS
#ifdef RBUILD
        Rcpp:Rcout << "Successful build." << std::endl;
#else
        std::cerr << "Successful build." << std::endl;
#endif
#endif

#ifdef DOUBLE_CHECK
#ifdef RBUILD
        Rcpp::Rcout << kernelSumOfSquaredResidualsVector.get_program().source() << std::endl;
#else
        std::cerr << kernelSumOfSquaredResidualsVector.get_program().source() << std::endl;
#endif
//        exit(-1);
#endif // DOUBLE_CHECK

		size_t index = 0;
		kernelSumOfSquaredResidualsVector.set_arg(index++, dLocations0); // Must update
		kernelSumOfSquaredResidualsVector.set_arg(index++, dObservations);
		kernelSumOfSquaredResidualsVector.set_arg(index++, dSquaredResiduals);

		if (isLeftTruncated) {
// 			kernelSumOfSquaredResidualsVector.set_arg(index++, dTruncations);
			kernelSumOfSquaredResidualsVector.set_arg(index++, static_cast<RealType>(precision)); // Must update
			kernelSumOfSquaredResidualsVector.set_arg(index++, static_cast<RealType>(oneOverSd)); // Must update
		}
		kernelSumOfSquaredResidualsVector.set_arg(index++, boost::compute::uint_(locationCount));

	}

	void createOpenCLGradientKernel() {

        const char cdfString1Double[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                static double cdf(double);

                static double cdf(double value) {
                    return 0.5 * erfc(-value * M_SQRT1_2);
                }
        );

        const char pdfString1Double[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                static double pdf(double);

                static double pdf(double value) {
                    return 0.5 * M_SQRT1_2 * M_2_SQRTPI * exp( - pow(value,2.0) * 0.5);
                }
        );

        const char cdfString1Float[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                static float cdf(float);

                static float cdf(float value) {

                    const float rSqrt2f = 0.70710678118655f;
                    return 0.5f * erfc(-value * rSqrt2f);
                }
        );

        const char pdfString1Float[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                static float pdf(float);

                static float pdf(float value) {

                    const float rSqrt2f = 0.70710678118655f;
                    const float rSqrtPif = 0.56418958354775f;
                    return rSqrt2f * rSqrtPif * exp( - pow(value,2.0f) * 0.5f);
                }
        );

		std::stringstream code;
		std::stringstream options;

		options << "-DTILE_DIM_I=" << TILE_DIM_I << " -DTPB=" << TPB << " -DDELTA=" << DELTA;

		if (sizeof(RealType) == 8) { // 64-bit fp
			code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
			options << " -DREAL=double -DREAL_VECTOR=double" << OpenCLRealType::dim  << " -DCAST=long"
                    << " -DZERO=0.0 -DONE=1.0 -DHALF=0.5";

			if (isLeftTruncated) {
				code << cdfString1Double;
                code << pdfString1Double;
			}

		} else { // 32-bit fp
			options << " -DREAL=float -DREAL_VECTOR=float" << OpenCLRealType::dim << " -DCAST=int"
                    << " -DZERO=0.0f -DONE=1.0f -DHALF=0.5f";

			if (isLeftTruncated) {
				code << cdfString1Float;
				code << pdfString1Float;
			}
		}

#ifndef USE_VECTOR
		bool isNvidia = false; // TODO Check device name
#endif

		code <<
			 " __kernel void computeGradient(__global const REAL_VECTOR *locations,  \n" <<
			 "                               __global const REAL *observations,      \n" <<
			 "                               __global REAL_VECTOR *output,           \n" <<
		     "                               const REAL precision,                   \n" <<
			 "                               const uint locationCount) {             \n" <<
			 "                                                                       \n" <<
//			 "   const uint i = get_local_id(1) + get_group_id(0) * TILE_DIM_I;      \n" <<
			 "   const uint i = get_group_id(0);                                     \n" <<
//			 "   const int inBounds = (i < locationCount);                           \n" <<
			 "                                                                       \n" <<
			 "   const uint lid = get_local_id(0);                                   \n" <<
			 "   uint j = get_local_id(0);                                           \n" <<
			 "                                                                       \n" <<
			 "   __local REAL_VECTOR scratch[TPB];                                   \n" <<
			 "                                                                       \n" <<
			 "   const REAL_VECTOR vectorI = locations[i];                           \n" <<
			 "   REAL_VECTOR sum = ZERO;                                             \n" <<
			 "                                                                       \n" <<
			 "   while (j < locationCount) {                                         \n" <<
			 "                                                                       \n" <<
			 "     const REAL_VECTOR vectorJ = locations[j];                         \n" <<
             "     const REAL_VECTOR difference = vectorI - vectorJ;                 \n";

        if (OpenCLRealType::dim == 8) {
            code << "     const REAL distance = sqrt(                                \n" <<
                    "              dot(difference.lo, difference.lo) +               \n" <<
                    "              dot(difference.hi, difference.hi)                 \n" <<
                    "      );                                                        \n";

        } else {
            code << "     const REAL distance = length(difference);                  \n";
        }

        // TODO Handle missing values by  `!isnan(observation) * `

        code <<
             "     const REAL observation = observations[i * locationCount + j];     \n" <<
             "     REAL residual = select(observation - distance, ZERO,              \n" <<
             "                                  (CAST)isnan(observation));           \n";

        if (isLeftTruncated) {
            code << "     const REAL trncDrv = select(-ONE / sqrt(precision) *        \n" << // TODO speed up this part
                    "                              pdf(distance * sqrt(precision)) / \n" <<
                    "                              cdf(distance * sqrt(precision)),  \n" <<
                    "                                 ZERO,                          \n" <<
                    "                                 (CAST)isnan(observation));     \n" <<
                    "     residual += trncDrv;                                       \n";
        }

        code <<
             "     REAL contrib = residual * precision / distance;                   \n" <<
			 "                                                                       \n" <<
             "     if (i != j) { sum += (vectorI - vectorJ) * contrib * DELTA;  }    \n" <<
			 "                                                                       \n" <<
			 "     j += TPB;                                                         \n" <<
			 "   }                                                                   \n" <<
			 "                                                                       \n" <<
			 "   scratch[lid] = sum;                                                 \n";
#ifdef USE_VECTOR
			 code << reduce::ReduceBody1<RealType,false>::body(); // TODO Try NVIDIA version at some point
#else
		code << (isNvidia ? reduce::ReduceBody2<RealType,true>::body() : reduce::ReduceBody2<RealType,false>::body());
#endif
		code <<
			 "   barrier(CLK_LOCAL_MEM_FENCE);                                       \n" <<
			 "   if (lid == 0) {                                                     \n";

        code <<
             "     REAL_VECTOR mask = (REAL_VECTOR) (";

        for (int i = 0; i < embeddingDimension; ++i) {
            code << " ONE";
            if (i < (OpenCLRealType::dim - 1)) {
                code << ",";
            }
        }
        for (int i = embeddingDimension; i < OpenCLRealType::dim; ++i) {
            code << " ZERO";
            if (i < (OpenCLRealType::dim - 1)) {
                code << ",";
            }
        }
        code << " ); \n";

        code <<
			 "     output[i] = mask * scratch[0];                                    \n" <<
			 "   }                                                                   \n" <<
			 " }                                                                     \n ";

#ifdef DOUBLE_CHECK_GRADIENT
#ifndef RBUILD
		std::cerr << code.str() << std::endl;
#endif
//        exit(-1);
#endif

#ifdef DEBUG_KERNELS
#ifndef RBUILD
        std::cerr << "Build gradient\n" << code.str() << std::endl;
#endif
#endif

		program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
		kernelGradientVector = boost::compute::kernel(program, "computeGradient");

#ifdef DEBUG_KERNELS
#ifndef RBUILD
        std::cerr << "Success" << std::endl;
#endif
#endif

		kernelGradientVector.set_arg(0, dLocations0); // Must update
		kernelGradientVector.set_arg(1, dObservations);
		kernelGradientVector.set_arg(2, dGradient);
		kernelGradientVector.set_arg(3, static_cast<RealType>(precision)); // Must update
		kernelGradientVector.set_arg(4, boost::compute::uint_(locationCount));
	}

	void createOpenCLKernels() {

		createOpenCLLikelihoodKernel();
		createOpenCLGradientKernel();

#ifdef DOUBLE_CHECK
 		using namespace boost::compute;
        boost::shared_ptr<program_cache> cache = program_cache::get_global_cache(ctx);

		RealType sum = RealType(0.0);
		boost::compute::reduce(dSquaredResiduals.begin(), dSquaredResiduals.end(), &sum, queue);

		auto programInfo = *begin(cache->get_keys());
#ifndef RBUILD
		std::cerr << "Try " << programInfo.first << " : " << programInfo.second << std::endl;
#endif
        boost::compute::program programReduce = *cache->get(programInfo.first, programInfo.second);
        auto kernelReduce = kernel(programReduce, "reduce");
#ifndef RBUILD
        std::cerr << programReduce.source() << std::endl;
#endif

        const auto &device2 = queue.get_device();
#ifndef RBUILD
        std::cerr << "nvidia? " << detail::is_nvidia_device(device) << " " << device.name() << " " << device.vendor() << std::endl;
        std::cerr << "nvidia? " << detail::is_nvidia_device(device2) << " " << device2.name() << " " << device.vendor() << std::endl;

		std::cerr << "Done compile VECTOR." << std::endl;
#endif
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

	mm::MemoryManager<RealType> gradient;

    mm::GPUMemoryManager<RealType> dObservations;

#ifdef USE_VECTORS
    mm::GPUMemoryManager<VectorType> dLocations0;
    mm::GPUMemoryManager<VectorType> dLocations1;

    mm::GPUMemoryManager<VectorType>* dLocationsPtr;
    mm::GPUMemoryManager<VectorType>* dStoredLocationsPtr;

	mm::GPUMemoryManager<VectorType> dGradient;
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
    mm::MemoryManager<double> doubleBuffer;

    boost::compute::program program;

#ifdef USE_VECTORS
	boost::compute::kernel kernelSumOfSquaredResidualsVector;
	boost::compute::kernel kernelGradientVector;
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

#endif // _OPENCL_MULTIDIMENSIONAL_SCALING_HPP
