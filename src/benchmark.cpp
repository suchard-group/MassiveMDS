
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>
#include <tbb/task_scheduler_init.h>

#include "AbstractMultiDimensionalScaling.hpp"


int cnt = 0;

template <typename T, typename PRNG, typename D>
void generateLocation(T& locations, D& d, PRNG& prng) {
    //int i = cnt++;
	for (auto& location : locations) {
		location = d(prng);
//        location = i++;
	}
}

//double getGradient

int main(int argc, char* argv[]) {

	// Set-up CLI
	namespace po = boost::program_options;
	po::options_description desc("Allowed options");
	desc.add_options()
            ("help", "produce help message")
            ("gpu", po::value<int>()->default_value(0), "number of GPU on which to run")
            ("tbb", po::value<int>()->default_value(0), "use TBB with specified number of threads")
            ("float", "run in single-precision")
            ("truncation", "enable truncation")
            ("iterations", po::value<int>()->default_value(1), "number of iterations")
            ("locations", po::value<int>()->default_value(3), "number of locations")
            ("dimension", po::value<int>()->default_value(2), "number of dimensions")
			("internal", "use internal dimension")
			("missing", "allow for missing entries")
            ("sse", "use hand-rolled SSE")
            ("avx", "use hand-rolled AVX")
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
	auto prng2 = std::mt19937(seed);

	std::cout << "Loading data" << std::endl;

	int embeddingDimension = vm["dimension"].as<int>();
	int locationCount = vm["locations"].as<int>();
	
	bool updateAllLocations = true;

	long flags = 0L;

	auto normal = std::normal_distribution<double>(0.0, 1.0);
	auto uniform = std::uniform_int_distribution<int>(0, locationCount - 1);
	auto binomial = std::bernoulli_distribution(0.75);
	auto normalData = std::normal_distribution<double>(0.0, 1.0);
	auto toss = std::bernoulli_distribution(0.25);
	
	std::shared_ptr<tbb::task_scheduler_init> task{nullptr};

    int deviceNumber = -1;
    int threads = 0;
	if (vm["gpu"].as<int>() > 0) {
		std::cout << "Running on GPU" << std::endl;
		flags |= mds::Flags::OPENCL;
        deviceNumber = vm["gpu"].as<int>() - 1;
	} else {
		std::cout << "Running on CPU" << std::endl;
		
		threads = vm["tbb"].as<int>();
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

	if (vm.count("sse") || vm.count("avx")) {
#ifndef USE_SIMD
        std::cerr << "SIMD is not implemented" << std::endl;
        exit(-1);
#endif
        if (vm.count("sse")) {
            if (vm.count("avx")) {
                std::cerr << "Can not request SSE and AVX simultaneously" << std::endl;
                exit(-1);
            }
            flags |= mds::Flags::SSE;
        } else {
            flags |= mds::Flags::AVX;
        }
	}
	
	bool truncation = false;
	if (vm.count("truncation")) {
		std::cout << "Enabling truncation" << std::endl;
		flags |= mds::Flags::LEFT_TRUNCATION;		
		truncation = true;
	}

	bool internalDimension = vm.count("internal");

	mds::SharedPtr instance = mds::factory(embeddingDimension, locationCount, flags, deviceNumber, threads);

	bool missing = vm.count("missing");
	if (missing) {
        std::cout << "Allowing for missingness" << std::endl;
	}

	auto elementCount = locationCount * locationCount;
	std::vector<double> data(elementCount);
	for (int i = 0; i < locationCount; ++i) {
	    data[i * locationCount + i] = 0.0;
	    for (int j = i + 1; j < locationCount; ++j) {

	        const double draw = normalData(prng);
	        double distance = draw * draw;

	        if (missing && toss(prng2)) {
	            distance = NAN;
	        }
	        data[i * locationCount + j] = distance;
	        data[j * locationCount + i] = distance;
	    }
	}

//	if (vm.count("missing")) {
//
//        data[2*locationCount + 3] = NAN;
//		data[3*locationCount + 2] = NAN;
//
//		for (int m = 3; m < 100; ++m) {
//		    for (int n = 5; n < 8; ++n) {
//		        data[m * locationCount + n] = data[n * locationCount + m] = NAN;
//		    }
//		}
//	}

	instance->setPairwiseData(&data[0], elementCount);

	int dataDimension = internalDimension ? instance->getInternalDimension() : embeddingDimension;

	std::vector<double> location(dataDimension);
	std::vector<double> allLocations;
	if (updateAllLocations) {
		allLocations.resize(dataDimension * locationCount);
	}

	double total = 0.0;
	for (int i = 0; i < locationCount; ++i) {
		generateLocation(location, normal, prng);
		instance->updateLocations(i, &location[0], dataDimension);
// 		for (int j = 0; j < embeddingDimension; ++j) {
// 			total += location[j];
// 		}
	}
// 	std::cerr << "FIND: " << total << std::endl;

    int gradientIndex = 1;

	double precision = 1.0;
	instance->setParameters(&precision, 1);

	instance->makeDirty();
	auto logLik = instance->getSumOfIncrements();

    std::vector<double> gradient(locationCount * dataDimension);

    instance->getLogLikelihoodGradient(gradient.data(), locationCount * dataDimension);
//    double sumGradient = std::accumulate(std::begin(gradient), std::end(gradient), 0.0);
    double sumGradient = gradient[gradientIndex];

// 	double logTrunc = 0.0;
// 	if (truncation) {
// 		logTrunc = instance->getSumOfLogTruncations();
// 	}

	std::cout << "Starting MDS benchmark" << std::endl;
	auto startTime = std::chrono::steady_clock::now();

	int iterations = vm["iterations"].as<int>();
	
	double timer = 0;
    double timer2 = 0;

	for (auto itr = 0; itr < iterations; ++itr) {

// 	    double startDiagnostic = instance->getDiagnostic();

		instance->storeState();

		if (updateAllLocations) {
			generateLocation(allLocations, normal, prng);
			instance->updateLocations(-1, &allLocations[0], dataDimension * locationCount);
		} else {
			int dimension = uniform(prng);
			generateLocation(location, normal, prng);
			instance->updateLocations(dimension, &location[0], dataDimension);
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

        auto startTime2 = std::chrono::steady_clock::now();

        instance->getLogLikelihoodGradient(gradient.data(), locationCount * dataDimension);

        auto duration2 = std::chrono::steady_clock::now() - startTime2;
        timer2 += std::chrono::duration<double, std::milli>(duration2).count();

//        sumGradient += std::accumulate(std::begin(gradient), std::end(gradient), 0.0) + 1;
        sumGradient += gradient[gradientIndex];

	}
	logLik /= iterations + 1;
    sumGradient /= iterations + 1;
// 	logTrunc /= iterations + 1;

	auto endTime = std::chrono::steady_clock::now();
	auto duration = endTime - startTime;

	std::cout << "End MDS benchmark" << std::endl;
	std::cout << "AvgLogLik = " << logLik << std::endl;
    std::cout << "AvgSumGradient = " << sumGradient << std::endl;
// 	std::cout << "AveLogTru = " << logTrunc << std::endl;
	std::cout << timer  << " ms" << std::endl;
    std::cout << timer2 << " ms" << std::endl;

	std::cout << std::chrono::duration<double, std::milli> (duration).count() << " ms "
			  << std::endl;

	std::ofstream outfile;
	outfile.open("report.txt",std::ios_base::app);
    outfile << deviceNumber << " " << threads << " " << locationCount << " " << embeddingDimension << " " << iterations << " " << timer << " " << timer2 << "\n" ;
	outfile.close();

}
