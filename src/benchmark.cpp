
#include <chrono>
#include <random>
#include <iostream>

#include <boost/program_options.hpp>
#include <tbb/task_scheduler_init.h>

#include "AbstractMultiDimensionalScaling.hpp"

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
		("tbb", po::value<int>()->default_value(0), "use TBB with specified number of threads")
		("float", "run in single-precision")
		("truncation", "enable truncation")
		("iterations", po::value<int>()->default_value(10), "number of iterations")
		("locations", po::value<int>()->default_value(6000), "number of locations")
	;
	po::variables_map vm;

	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
	} catch (std::exception& e) {
		std::cout << desc << std::endl;
		return 1;
	}

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}

	long seed = 666L;

	auto prng = std::mt19937(seed);

	std::cout << "Loading data" << std::endl;

	int embeddingDimension = 2;
	int locationCount = vm["locations"].as<int>();
	
	bool updateAllLocations = true;

	long flags = 0L;

	auto normal = std::normal_distribution<double>(0.0, 1.0);
	auto uniform = std::uniform_int_distribution<int>(0, locationCount - 1);
	auto binomial = std::bernoulli_distribution(0.75);
	auto normalData = std::normal_distribution<double>(0.0, 1.0);
	
	std::shared_ptr<tbb::task_scheduler_init> task{nullptr};

	if (vm.count("gpu")) {
		std::cout << "Running on GPU" << std::endl;
		flags |= mds::Flags::OPENCL;
	} else {
		std::cout << "Running on CPU" << std::endl;
		
		int threads = vm["tbb"].as<int>();
		if (threads != 0) {
			std::cout << "Using TBB with " << threads << " out of " 
			          << tbb::task_scheduler_init::default_num_threads()
			          << " threads" << std::endl;
			flags |= mds::Flags::TBB;
			task = std::make_shared<tbb::task_scheduler_init>(threads);
		}
	}
	
	if (vm.count("float")) {
		std::cout << "Running in single-precision" << std::endl;
		flags |= mds::Flags::FLOAT;
	} else {
		std::cout << "Running in double-precision" << std::endl;
	}
	
	bool truncation = false;
	if (vm.count("truncation")) {
		std::cout << "Enabling truncation" << std::endl;
		flags |= mds::Flags::LEFT_TRUNCATION;		
		truncation = true;
	}

	mds::SharedPtr instance = mds::factory(embeddingDimension, locationCount, flags);

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
// 		for (int j = 0; j < embeddingDimension; ++j) {
// 			total += location[j];
// 		}
	}
// 	std::cerr << "FIND: " << total << std::endl;

	double precision = 1.0;
	instance->setParameters(&precision, 1);

	instance->makeDirty();
	auto logLik = instance->getSumOfIncrements();
	
// 	double logTrunc = 0.0;
// 	if (truncation) {
// 		logTrunc = instance->getSumOfLogTruncations();
// 	}

	std::cout << "Starting MDS benchmark" << std::endl;
	auto startTime = std::chrono::steady_clock::now();

	int iterations = vm["iterations"].as<int>();
	
	double timer = 0;

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
		
		auto startTime1 = std::chrono::steady_clock::now();

		double inc = instance->getSumOfIncrements();
		logLik += inc;
		
// 		if (truncation) {
// 			double trunc = instance->getSumOfLogTruncations();
// 			logTrunc += trunc;
// 		}
		
		auto duration1 = std::chrono::steady_clock::now() - startTime1;
		timer += std::chrono::duration<double, std::milli>(duration1).count();

		bool restore = binomial(prng);
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
// 	logTrunc /= iterations + 1;

	auto endTime = std::chrono::steady_clock::now();
	auto duration = endTime - startTime;

	std::cout << "End MDS benchmark" << std::endl;
	std::cout << "AvgLogLik = " << logLik << std::endl;
// 	std::cout << "AveLogTru = " << logTrunc << std::endl;
	std::cout << timer << " ms" << std::endl;
	std::cout << std::chrono::duration<double, std::milli> (duration).count() << " ms "
			  << std::endl;

	std::vector<double> gradient(locationCount * embeddingDimension);
	instance->getLogLikelihoodGradient(gradient);

}
