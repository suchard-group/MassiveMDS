#ifndef _OPENCL_MULTIDIMENSIONAL_SCALING_HPP
#define _OPENCL_MULTIDIMENSIONAL_SCALING_HPP

// Turn off warning about Atomics from older BOOST
#if defined(__clang__)
# pragma clang diagnostic push
#endif

#if defined(__clang__) && defined(__has_warning)
# if __has_warning( "-Wc11-extensions" )
#  pragma clang diagnostic ignored "-Wc11-extensions"
# endif
#endif

#include <iostream>
#include <cmath>

#include "AbstractMultiDimensionalScaling.hpp"

#include <boost/compute/algorithm/reduce.hpp>
#include "reduce_fast.hpp"

#define TILE_DIM 16
#define TILE_DIM_I  128
#define TPB 128
#define DELTA 1

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

    OpenCLMultiDimensionalScaling(int embeddingDimension, Layout layout, long flags, int deviceNumber)
        : AbstractMultiDimensionalScaling(embeddingDimension, layout, flags),
          precision(0.0), storedPrecision(0.0),
          oneOverSd(0.0), storedOneOverSd(0.0),
          sumOfSquaredResiduals(0.0), storedSumOfSquaredResiduals(0.0),
          sumOfTruncations(0.0), storedSumOfTruncations(0.0),

          observations(layout.observationCount),
          transposedData(layout.isSymmetric() ? 0 : layout.observationCount),

          locations0(layout.uniqueLocationCount * OpenCLRealType::dim),
		  locations1(layout.uniqueLocationCount * OpenCLRealType::dim),
		  locationsPtr(&locations0),
		  storedLocationsPtr(&locations1),

          squaredResiduals(layout.observationCount),
          storedSquaredResiduals(layout.uniqueLocationCount),

          isStoredSquaredResidualsEmpty(false),
          isStoredTruncationsEmpty(false)
    {
      MDS_CERR << "ctor OpenCLMultiDimensionalScaling" << std::endl;

      MDS_CERR << "All devices:" << std::endl;

      const auto devices = boost::compute::system::devices();

      for (const auto &dev : devices) {
          MDS_CERR << "\t" << dev.name() << std::endl;
      }

      if (deviceNumber < 0 || deviceNumber >= devices.size()) {
        device = boost::compute::system::default_device();
      } else {
        device = devices[deviceNumber];
      }

      if (device.type() != CL_DEVICE_TYPE_GPU) {
          MDS_STOP("Error: selected device is not a GPU.\n");
      } else {
          MDS_CERR << "Using: " << device.name() << " with vector-dim = " << OpenCLRealType::dim << std::endl;
      }

      if ((sizeof(RealType) == 8) && !device.supports_extension("cl_khr_fp64")) {
          MDS_STOP("Error: device does not support FP64.\n");
      }

      ctx = boost::compute::context(device, nullptr);
      queue = boost::compute::command_queue{ctx, device
        , boost::compute::command_queue::enable_profiling
      };

      dObservations = mm::GPUMemoryManager<RealType>(observations.size(), ctx);
      if (!layout.isSymmetric()) {
          dTransposedObservations = mm::GPUMemoryManager<RealType>(observations.size(), ctx);
      }

		dLocations0 = mm::GPUMemoryManager<VectorType>(layout.uniqueLocationCount, ctx);
		dLocations1 = mm::GPUMemoryManager<VectorType>(layout.uniqueLocationCount, ctx);
		dGradient   = mm::GPUMemoryManager<VectorType>(layout.uniqueLocationCount, ctx);

		dLocationsPtr = &dLocations0;
		dStoredLocationsPtr = &dLocations1;

		dSquaredResiduals = mm::GPUMemoryManager<RealType>(squaredResiduals.size(), ctx);
		dStoredSquaredResiduals = mm::GPUMemoryManager<RealType>(storedSquaredResiduals.size(), ctx);

    	if (flags & Flags::LEFT_TRUNCATION) {
    		isLeftTruncated = true;

    		MDS_COUT << "Using left truncation" << std::endl;

    		truncations.resize(layout.observationCount);
    		storedTruncations.resize(layout.uniqueLocationCount);

    		dTruncations = mm::GPUMemoryManager<RealType>(truncations.size(), ctx);
    		dStoredTruncations = mm::GPUMemoryManager<RealType>(storedTruncations.size(), ctx);
    	}

		createOpenCLKernels();
    }

    void updateLocations(int locationIndex, double* location, size_t length) override {

		size_t offset{0};
		size_t deviceOffset{0};

		if (locationIndex == -1) {
			// Update all locations
			assert(length == OpenCLRealType::dim * layout.uniqueLocationCount ||
                    length == embeddingDimension * layout.uniqueLocationCount);

			incrementsKnown = false;
			isStoredSquaredResidualsEmpty = true;
			isStoredTruncationsEmpty = true;

		} else {
			// Update a single location
    		assert(length == OpenCLRealType::dim ||
                    length == embeddingDimension);

	    	if (updatedLocation != - 1) {
    			// more than one location updated -- do a full re-computation
	    		incrementsKnown = false;
	    		isStoredSquaredResidualsEmpty = true;
	    		isStoredTruncationsEmpty = true;
    		}

	    	updatedLocation = locationIndex;

			offset = locationIndex * OpenCLRealType::dim;
	    	deviceOffset = static_cast<size_t>(locationIndex);
	    }

        // If requires padding
        if (embeddingDimension != OpenCLRealType::dim) {
            if (locationIndex == -1) {

                mm::paddedBufferedCopy(location, embeddingDimension, embeddingDimension,
                                       begin(*locationsPtr) + offset, OpenCLRealType::dim,
                                       layout.uniqueLocationCount, buffer);

                length = OpenCLRealType::dim * layout.uniqueLocationCount; // New padded length

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

		kernelGradientVector.set_arg(0, *dLocationsPtr);
		kernelGradientVector.set_arg(3, static_cast<RealType>(precision));

        using uint_ = boost::compute::uint_;

        if (!layout.isSymmetric()) {
            // Row locations first
            kernelGradientVector.set_arg(1, dObservations);
            kernelGradientVector.set_arg(4, uint_(layout.rowLocationCount));
            kernelGradientVector.set_arg(5, uint_(layout.columnLocationCount));
            kernelGradientVector.set_arg(6, uint_(0));
            kernelGradientVector.set_arg(7, uint_(layout.columnLocationOffset));
            kernelGradientVector.set_arg(8, uint_(0));
        }

		queue.enqueue_1d_range_kernel(kernelGradientVector, 0,
                                      static_cast<unsigned int>(layout.rowLocationCount) * TPB, TPB);

        if (!layout.isSymmetric()) {
            // Column locations second
            kernelGradientVector.set_arg(1, dTransposedObservations);
            kernelGradientVector.set_arg(4, uint_(layout.columnLocationCount));
            kernelGradientVector.set_arg(5, uint_(layout.rowLocationCount));
            kernelGradientVector.set_arg(6, uint_(layout.columnLocationOffset));
            kernelGradientVector.set_arg(7, uint_(0));
            kernelGradientVector.set_arg(8, uint_(layout.rowLocationCount));

            queue.enqueue_1d_range_kernel(kernelGradientVector, 0,
                                          static_cast<unsigned int>(layout.columnLocationCount) * TPB, TPB);
        }
        queue.finish();

        if (length == layout.uniqueLocationCount * OpenCLRealType::dim) {

            mm::bufferedCopyFromDevice<OpenCLRealType>(dGradient.begin(), dGradient.end(),
                                       result, buffer, queue);
            queue.finish();

        } else {

            if (doubleBuffer.size() != layout.uniqueLocationCount * OpenCLRealType::dim) {
                doubleBuffer.resize(layout.uniqueLocationCount * OpenCLRealType::dim);
            }

            mm::bufferedCopyFromDevice<OpenCLRealType>(dGradient.begin(), dGradient.end(),
                                       doubleBuffer.data(), buffer, queue);
            queue.finish();

            mm::paddedBufferedCopy(begin(doubleBuffer), OpenCLRealType::dim, embeddingDimension,
                                   result, embeddingDimension,
                                   layout.uniqueLocationCount, buffer);
        }
 	}

    void computeResidualsAndTruncations() {

		if (!incrementsKnown) {
            computeSumOfSquaredResiduals();
			incrementsKnown = true;
		} else {
			MDS_CERR << "SHOULD NOT BE HERE" << std::endl;
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

#if 0
 	double getSumOfLogTruncations() {
    	if (!sumOfIncrementsKnown) {
			computeResidualsAndTruncations();
			sumOfIncrementsKnown = true;
		}
 		return sumOfTruncations;
 	}
#endif

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

    void acceptState() override {
        if (!isStoredSquaredResidualsEmpty) {
            const int cnt = layout.uniqueLocationCount;
    		for (int j = 0; j < cnt; ++j) {
    			squaredResiduals[j * cnt + updatedLocation] = squaredResiduals[updatedLocation * cnt + j];
    		}

    		// COMPUTE TODO

    		if (isLeftTruncated) {
                for (int j = 0; j < cnt; ++j) {
	    			truncations[j * cnt + updatedLocation] = truncations[updatedLocation * cnt + j];
	    		}

	    		// COMPUTE TODO
    		}
    	}
    }

    void restoreState() override {
    	sumOfSquaredResiduals = storedSumOfSquaredResiduals;
    	sumOfIncrementsKnown = true;

		if (!isStoredSquaredResidualsEmpty) {
            const int cnt = layout.uniqueLocationCount;
    		std::copy(
    			begin(storedSquaredResiduals),
    			end(storedSquaredResiduals),
    			begin(squaredResiduals) + updatedLocation * cnt
    		);

    		// COMPUTE
    		boost::compute::copy(
                    dStoredSquaredResiduals.begin(),
                    dStoredSquaredResiduals.end(),
    			dSquaredResiduals.begin() + updatedLocation * cnt, queue
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
	    			begin(truncations) + updatedLocation * count
	    		);

	    		// COMPUTE
	    		boost::compute::copy(
	    			dStoredTruncations.begin(),
	    			dStoredTruncations.end(),
	    			dTruncations.begin() + updatedLocation * count, queue
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

    // TODO use layout.observationStride

    void setPairwiseData(double* data, size_t length) override {

		assert(length == observations.size());

        if (layout.isSymmetric()) {
            for (int i = 0; i < layout.rowLocationCount; ++i) {
                data[i * layout.observationStride + i] = std::nan("");
            }
        }

		mm::bufferedCopy(data, data + length, begin(observations), buffer);

		// COMPUTE
		mm::bufferedCopyToDevice(data, data + length, dObservations.begin(),
                                 buffer, queue);

        if (!layout.isSymmetric()) {
            for (int i = 0; i < layout.rowLocationCount; ++i) {
                for (int j = 0; j < layout.columnLocationCount; ++j) {
                    transposedData[j * layout.rowLocationCount + i] =
                            data[i * layout.columnLocationCount + j];
                }
            }

            // COMPUTE
            double* transposed = &(transposedData[0]);
            mm::bufferedCopyToDevice(transposed, transposed + length,
                                     dTransposedObservations.begin(), buffer, queue);
        }
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

	void computeSumOfSquaredResiduals() {

		// COMPUTE TODO
		auto startTime2 = std::chrono::steady_clock::now();

        const int rowCount = layout.rowLocationCount;
        const int colCount = layout.columnLocationCount;

		kernelSumOfSquaredResidualsVector.set_arg(0, *dLocationsPtr);

		if (isLeftTruncated) {
			kernelSumOfSquaredResidualsVector.set_arg(7, static_cast<RealType>(precision));
			kernelSumOfSquaredResidualsVector.set_arg(8, static_cast<RealType>(oneOverSd));
		}

		const size_t local_work_size[2] = {TILE_DIM, TILE_DIM};
		size_t row_work_groups = rowCount / TILE_DIM;
		if (rowCount % TILE_DIM != 0) {
			++row_work_groups;
		}
        size_t col_work_groups = colCount / TILE_DIM;
        if (colCount % TILE_DIM != 0) {
            ++col_work_groups;
        }
		const size_t global_work_size[2] = {col_work_groups * TILE_DIM, row_work_groups * TILE_DIM};

		queue.enqueue_nd_range_kernel(kernelSumOfSquaredResidualsVector, 2, nullptr, global_work_size, local_work_size);

		queue.finish();
		auto duration2 = std::chrono::steady_clock::now() - startTime2;
		if (count > 1) timer2 += std::chrono::duration<double, std::milli>(duration2).count();


        auto startTime3 = std::chrono::steady_clock::now();
		RealType sum = RealType(0.0);
		boost::compute::reduce(dSquaredResiduals.begin(), dSquaredResiduals.end(), &sum, queue);

		queue.finish();
		auto duration3 = std::chrono::steady_clock::now() - startTime3;
		if (count > 1) timer3 += std::chrono::duration<double, std::milli>(duration3).count();

//  		using namespace boost::compute;
//         boost::shared_ptr<program_cache> cache = program_cache::get_global_cache(ctx);
//
//         auto list = cache->get_keys();
//         for (auto x : list) {
//             MDS_CERR << x.first << " " << x.second << std::endl;
//             MDS_CERR << cache->get(x.first, x.second)->source() << std::endl;
//         }
//         exit(-1);

    	sumOfSquaredResiduals = sum;

	    incrementsKnown = true;
	    sumOfIncrementsKnown = true;

	    count++;
	}

	void updateSumOfSquaredResiduals() {

        MDS_CERR << "Not yet implemented" << std::endl;
        exit(-1);

#if 0
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
#endif
	}

	void updateSumOfSquaredResidualsAndTruncations() {

        MDS_CERR << "Not yet implemented." << std::endl;
        exit(-1);

#if 0
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
#endif
	}

	void createOpenCLLikelihoodKernel() {

		const char cdfString1Double[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                __attribute__((unused)) static double cdf(double);

                __attribute__((unused)) static double cdf(double value) {
	    		return 0.5 * erfc(-value * M_SQRT1_2);
	    	}
		);

		const char cdfString1Float[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                __attribute__((unused)) static float cdf(float);

                __attribute__((unused)) static float cdf(float value) {

				const float rSqrt2f = 0.70710678118655f;
	    		return 0.5f * erfc(-value * rSqrt2f);
	    	}
		);

		std::stringstream code;
		std::stringstream options;

		options << "-DTILE_DIM=" << TILE_DIM;

		if (sizeof(RealType) == 8) { // 64-bit fp
			code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
			options << " -DREAL=double -DREAL_VECTOR=double" << OpenCLRealType::dim << " -DCAST=long"
                    << " -DZERO=0.0 -DHALF=0.5";

			if (isLeftTruncated) {
				code << cdfString1Double << "\n";
			}

		} else { // 32-bit fp
			options << " -DREAL=float -DREAL_VECTOR=float" << OpenCLRealType::dim << " -DCAST=int"
                    << " -DZERO=0.0f -DHALF=0.5f";

			if (isLeftTruncated) {
				code << cdfString1Float << "\n";
			}
		}

		code <<
			" __kernel void computeSSR(__global const REAL_VECTOR *locations,  \n" << // 0
			"                          __global const REAL *observations,      \n" << // 1
			"                          __global REAL *squaredResiduals,        \n" << // 2
            "                          const uint rowLocationCount,            \n" << // 3
            "                          const uint colLocationCount,            \n" << // 4
            "                          const uint rowOffset,                   \n" << // 5
            "                          const uint colOffset";                         // 6

		if (isLeftTruncated) {
			code << ", \n" <<
            "                          const REAL precision,                   \n" << // 7
			"                          const REAL oneOverSd) {                 \n";   // 8
		} else {
            code << ") { \n";
        }

		code << BOOST_COMPUTE_STRINGIZE_SOURCE(
				const uint offsetJ = get_group_id(0) * TILE_DIM;
				const uint offsetI = get_group_id(1) * TILE_DIM;

				const uint j = offsetJ + get_local_id(0);
				const uint i = offsetI + get_local_id(1);

				__local REAL_VECTOR tile[2][TILE_DIM + 1]; // tile[0] == locations_j, tile[1] == locations_i

				if (get_local_id(1) < 2) { // load just 2 rows
					tile[get_local_id(1)][get_local_id(0)] = locations[
						(get_local_id(1) - 0) * (rowOffset + offsetI + get_local_id(0)) + // tile[1] = locations_i
						(1 - get_local_id(1)) * (colOffset + offsetJ + get_local_id(0))   // tile[0] = locations_j
					];
				}

				barrier(CLK_LOCAL_MEM_FENCE);

				if (i < rowLocationCount && j < colLocationCount) {

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
                    const REAL observation = observations[i * colLocationCount + j];
                    const REAL residual = select(distance - observation, ZERO, (CAST)isnan(observation));
                    REAL squaredResidual = residual * residual;
        );

		if (isLeftTruncated) {
			code << BOOST_COMPUTE_STRINGIZE_SOURCE(
					squaredResidual *= HALF * precision;
					const REAL truncation = select(log(cdf(fabs(distance) * oneOverSd)), ZERO, (CAST)isnan(observation));
					squaredResidual += truncation;
			);
		}

		code << BOOST_COMPUTE_STRINGIZE_SOURCE(
			 		squaredResiduals[i * colLocationCount + j] = squaredResidual;
				}
            }
		);

#ifdef DEBUG_KERNELS
        MDS_CERR << "Likelihood kernel\n" << code.str() << std::endl;
#endif

		program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
        kernelSumOfSquaredResidualsVector = boost::compute::kernel(program, "computeSSR");

#ifdef DEBUG_KERNELS
        MDS_CERR << "Successful build." << std::endl;
#endif

		kernelSumOfSquaredResidualsVector.set_arg(0, dLocations0); // Must update
		kernelSumOfSquaredResidualsVector.set_arg(1, dObservations);
		kernelSumOfSquaredResidualsVector.set_arg(2, dSquaredResiduals);

        using uint_ = boost::compute::uint_;

        kernelSumOfSquaredResidualsVector.set_arg(3, uint_(layout.rowLocationCount));  // 3
        kernelSumOfSquaredResidualsVector.set_arg(4, uint_(layout.columnLocationCount));
        kernelSumOfSquaredResidualsVector.set_arg(5, uint_(0));
        kernelSumOfSquaredResidualsVector.set_arg(6, uint_(layout.columnLocationOffset));

		if (isLeftTruncated) {
			kernelSumOfSquaredResidualsVector.set_arg(7, static_cast<RealType>(precision)); // 7 Must update
			kernelSumOfSquaredResidualsVector.set_arg(8, static_cast<RealType>(oneOverSd)); // 8 Must update
		}
	}

	void createOpenCLGradientKernel() {

        const char cdfString1Double[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                __attribute__((unused)) static double cdf(double);

                __attribute__((unused)) static double cdf(double value) {
                    return 0.5 * erfc(-value * M_SQRT1_2);
                }
        );

        const char pdfString1Double[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                __attribute__((unused)) static double pdf(double);

                __attribute__((unused)) static double pdf(double value) {
                    return 0.5 * M_SQRT1_2 * M_2_SQRTPI * exp( - pow(value,2.0) * 0.5);
                }
        );

        const char cdfString1Float[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                __attribute__((unused)) static float cdf(float);

                __attribute__((unused)) static float cdf(float value) {

                    const float rSqrt2f = 0.70710678118655f;
                    return 0.5f * erfc(-value * rSqrt2f);
                }
        );

        const char pdfString1Float[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                __attribute__((unused)) static float pdf(float);

                __attribute__((unused)) static float pdf(float value) {

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
				code << cdfString1Double << "\n";
                code << pdfString1Double << "\n";
			}

		} else { // 32-bit fp
			options << " -DREAL=float -DREAL_VECTOR=float" << OpenCLRealType::dim << " -DCAST=int"
                    << " -DZERO=0.0f -DONE=1.0f -DHALF=0.5f";

			if (isLeftTruncated) {
				code << cdfString1Float << "\n";
				code << pdfString1Float << "\n";
			}
		}

		code <<
			 " __kernel void computeGradient(__global const REAL_VECTOR *locations,  \n" <<
			 "                               __global const REAL *observations,      \n" <<
			 "                               __global REAL_VECTOR *output,           \n" <<
		     "                               const REAL precision,                   \n" <<
			 "                               const uint rowLocationCount,            \n" <<
             "                               const uint colLocationCount,            \n" <<
             "                               const uint rowOffset,                   \n" <<
             "                               const uint colOffset,                   \n" <<
             "                               const uint gradientOffset) {            \n" <<
			 "                                                                       \n" <<
			 "   const uint i = get_group_id(0);                                     \n" <<
			 "                                                                       \n" <<
			 "   const uint lid = get_local_id(0);                                   \n" <<
			 "   uint j = get_local_id(0);                                           \n" <<
			 "                                                                       \n" <<
			 "   __local REAL_VECTOR scratch[TPB];                                   \n" <<
			 "                                                                       \n" <<
			 "   const REAL_VECTOR vectorI = locations[rowOffset + i];               \n" <<
			 "   REAL_VECTOR sum = ZERO;                                             \n" <<
			 "                                                                       \n" <<
			 "   while (j < colLocationCount) {                                      \n" <<
			 "                                                                       \n" <<
			 "     const REAL_VECTOR vectorJ = locations[colOffset + j];             \n" <<
             "     const REAL_VECTOR difference = vectorI - vectorJ;                 \n";

        if (OpenCLRealType::dim == 8) {
            code <<
             "     const REAL distance = sqrt(                                       \n" <<
             "              dot(difference.lo, difference.lo) +                      \n" <<
             "              dot(difference.hi, difference.hi));                      \n";

        } else {
            code <<
             "     const REAL distance = length(difference);                         \n";
        }

        code <<
             "     const REAL observation = observations[i * colLocationCount + j];  \n" <<
             "     REAL residual = select(observation - distance, ZERO,              \n" <<
             "                                  (CAST)isnan(observation));           \n";

        if (isLeftTruncated) {
            code <<
             "     const REAL truncDrv = select(-ONE / sqrt(precision) *             \n" <<
             "                              pdf(distance * sqrt(precision)) /        \n" <<
             "                              cdf(distance * sqrt(precision)),         \n" <<
             "                                 ZERO,                                 \n" <<
             "                                 (CAST)isnan(observation));            \n" <<
             "     residual += truncDrv;                                             \n";
        }

        code <<
             "     REAL contrib = residual * precision / distance;                   \n" <<
             "     sum += (vectorI - vectorJ) * contrib * DELTA;                     \n" <<
			 "     j += TPB;                                                         \n" <<
			 "   }                                                                   \n" <<
			 "                                                                       \n" <<
			 "   scratch[lid] = sum;                                                 \n";

        code << reduce::ReduceBody1<RealType,false>::body();

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
			 "     output[gradientOffset + i] = mask * scratch[0];                   \n" <<
			 "   }                                                                   \n" <<
			 " }                                                                     \n ";

#ifdef DEBUG_KERNELS
        MDS_CERR << "Build gradient\n" << code.str() << std::endl;
#endif

		program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
		kernelGradientVector = boost::compute::kernel(program, "computeGradient");

#ifdef DEBUG_KERNELS
        MDS_CERR << "Success" << std::endl;
#endif

        using uint_ = boost::compute::uint_;

		kernelGradientVector.set_arg(0, dLocations0); // Must update
		kernelGradientVector.set_arg(1, dObservations);
		kernelGradientVector.set_arg(2, dGradient);
		kernelGradientVector.set_arg(3, static_cast<RealType>(precision)); // Must update
		kernelGradientVector.set_arg(4, uint_(layout.rowLocationCount));
        kernelGradientVector.set_arg(5, uint_(layout.columnLocationCount));
        kernelGradientVector.set_arg(6, uint_(0));
        kernelGradientVector.set_arg(7, uint_(layout.columnLocationOffset));
        kernelGradientVector.set_arg(8, uint_(0));
	}

	void createOpenCLKernels() {
		createOpenCLLikelihoodKernel();
		createOpenCLGradientKernel();
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
    std::vector<double> transposedData;

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
    mm::GPUMemoryManager<RealType> dTransposedObservations;

    mm::GPUMemoryManager<VectorType> dLocations0;
    mm::GPUMemoryManager<VectorType> dLocations1;

    mm::GPUMemoryManager<VectorType>* dLocationsPtr;
    mm::GPUMemoryManager<VectorType>* dStoredLocationsPtr;

	mm::GPUMemoryManager<VectorType> dGradient;

    mm::GPUMemoryManager<RealType> dSquaredResiduals;
    mm::GPUMemoryManager<RealType> dStoredSquaredResiduals;

    mm::GPUMemoryManager<RealType> dTruncations;
    mm::GPUMemoryManager<RealType> dStoredTruncations;

    bool isStoredSquaredResidualsEmpty;
    bool isStoredTruncationsEmpty;

    mm::MemoryManager<RealType> buffer;
    mm::MemoryManager<double> doubleBuffer;

    boost::compute::program program;

	boost::compute::kernel kernelSumOfSquaredResidualsVector;
	boost::compute::kernel kernelGradientVector;

	double timer2 = 0;
	double timer3 = 0;
};

} // namespace mds

#endif // _OPENCL_MULTIDIMENSIONAL_SCALING_HPP
