
#include <chrono>
#include <random>
#include <iostream>

#include <boost/program_options.hpp>

#include "AbstractMultiDimensionalScaling.hpp"
// #include "OpenCLMultiDimensionalScaling.hpp"
// #include "NewMultiDimensionalScaling.hpp"
// #include "MultiDimensionalScaling.hpp"

typedef std::shared_ptr<mds::AbstractMultiDimensionalScaling> SharedPtr; // TODO Move to AMDS.hpp

// forward reference
namespace mds {
	SharedPtr constructMultiDimensionalScalingDouble(int, int, long);
	SharedPtr constructNewMultiDimensionalScalingDouble(int, int, long);
	SharedPtr constructOpenCLMultiDimensionalScalingDouble(int, int, long);

	SharedPtr constructMultiDimensionalScalingFloat(int, int, long);
	SharedPtr constructNewMultiDimensionalScalingFloat(int, int, long);
	SharedPtr constructOpenCLMultiDimensionalScalingFloat(int, int, long);


SharedPtr factory(int dim1, int dim2, long flags) {
	bool useFloat = flags & mds::Flags::FLOAT;
	bool useOpenCL = flags & mds::Flags::OPENCL;

	if (useFloat) {
		if (useOpenCL) {
			return constructOpenCLMultiDimensionalScalingFloat(dim1, dim2, flags);
		} else {
			return constructNewMultiDimensionalScalingFloat(dim1, dim2, flags);
		}
	} else {
		if (useOpenCL) {
			return constructOpenCLMultiDimensionalScalingDouble(dim1, dim2, flags);
		} else {
			return constructNewMultiDimensionalScalingDouble(dim1, dim2, flags);
		}
	}
}

} // namespace mds

template <typename T, typename PRNG, typename D>
void generateLocation(T& locations, D& d, PRNG& prng) {
	for (auto& location : locations) {
		location = d(prng);
	}
}

int main(int argc, char* argv[]) {

	// Set-up CLI
	namespace po = boost::program_options;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("gpu", "run on first GPU")
		("iterations", po::value<int>()->default_value(10), "number of iterations")
	;
	po::variables_map vm;

	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
	} catch (std::exception& e) {
		std::cout << desc << std::endl;
	}

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}

	long seed = 666L;

	auto prng = std::mt19937(seed);

	std::cout << "Loading data" << std::endl;

	int embeddingDimension = 2;
	int locationCount = 6000; // 2000; // 6000;
	bool updateAllLocations = true;


	long flags = 0L;
// 	flags |= mds::Flags::LEFT_TRUNCATION;

	auto normal = std::normal_distribution<double>(0.0, 1.0);
	auto uniform = std::uniform_int_distribution<int>(0, locationCount - 1);
	auto binomial = std::bernoulli_distribution(0.75);
	auto normalData = std::normal_distribution<double>(0.0, 1.0);

// 	SharedPtr instance =
// 		mds::constructMultiDimensionalScalingDouble
// 	(embeddingDimension, locationCount, flags);

	//flags |= mds::Flags::FLOAT;

	if (vm.count("gpu")) {
		std::cout << "Running on GPU" << std::endl;
		flags |= mds::Flags::OPENCL;
	} else {
		std::cout << "Running on CPU" << std::endl;
	}

	SharedPtr instance = mds::factory(embeddingDimension, locationCount, flags);

	auto elementCount = locationCount * locationCount;
	std::vector<double> data(elementCount);
	for (int i = 0; i < locationCount; ++i) {
	    data[i * locationCount + i] = 0.0;
	    for (int j = i + 1; j < locationCount; ++j) {
	        const double draw = normalData(prng);
	        const double distance = draw * draw;
	        data[i * locationCount + j] = distance;
	        data[j * locationCount + i] = distance;
	    }
	}

	instance->setPairwiseData(&data[0], elementCount);

	std::vector<double> location(embeddingDimension);
	std::vector<double> allLocations;
	if (updateAllLocations) {
		allLocations.resize(embeddingDimension * locationCount);
	}

	double total = 0.0;
	for (int i = 0; i < locationCount; ++i) {
		generateLocation(location, normal, prng);
		instance->updateLocations(i, &location[0], embeddingDimension);
		for (int j = 0; j < embeddingDimension; ++j) {
			total += location[j];
		}
	}
	std::cerr << "FIND: " << total << std::endl;

	instance->makeDirty();
	auto logLik = instance->getSumOfSquaredResiduals();

	std::cout << "Starting MDS benchmark" << std::endl;
	auto startTime = std::chrono::steady_clock::now();

	int iterations = vm["iterations"].as<int>();

	for (auto itr = 0; itr < iterations; ++itr) {

// 	    double startDiagnostic = instance->getDiagnostic();

		instance->storeState();

		if (updateAllLocations) {
			generateLocation(allLocations, normal, prng);
			instance->updateLocations(-1, &allLocations[0], embeddingDimension * locationCount);
		} else {
			int dimension = uniform(prng);
			generateLocation(location, normal, prng);
			instance->updateLocations(dimension, &location[0], embeddingDimension);
		}

		double inc = instance->getSumOfSquaredResiduals();
		logLik += inc;

		bool restore = binomial(prng);
//  		restore = false;
		if (restore) {
			instance->restoreState();
		} else {
		    instance->acceptState();
		}

// 		double endDiagnostic = instance->getDiagnostic();
// 		if (restore && (startDiagnostic != endDiagnostic)) {
// 		    std::cerr << "Failed restore" << std::endl;
// 		    std::cerr << (endDiagnostic - startDiagnostic) << std::endl;
// 		    exit(-1);
// 		}
// 		if (!restore && (startDiagnostic == endDiagnostic)) {
// 		    std::cerr << "Failed accept" << std::endl;
// 		    exit(-1);
// 		}
//
// 		std::cerr << endDiagnostic << std::endl;
// 		if (itr > 100) exit(-1);
	}
	logLik /= iterations + 1;

	auto endTime = std::chrono::steady_clock::now();
	auto duration = endTime - startTime;

	std::cout << "End MDS benchmark" << std::endl;
	std::cout << "AvgLogLik = " << logLik << std::endl;
	std::cout << std::chrono::duration<double, std::milli> (duration).count() << " ms "
			  << std::endl;

}
