
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>

#include "cxxopts.hpp"

#ifdef USE_TBB
	#define TBB_PREVIEW_GLOBAL_CONTROL 1
	#include "tbb/global_control.h"
#endif

#include "AbstractMultiDimensionalScaling.hpp"

int cnt = 0;

template <typename T, typename PRNG, typename D>
void generateLocation(T& locations, D& d, PRNG& prng) {
  int i = 1;
	for (auto& location : locations) {
		location = d(prng);
//        location = i++;
	}
}

int main(int argc, char* argv[]) {

	// Set-up CLI

    cxxopts::Options options("benchmark2", "Benchmark row-by-column multidimensional scaling");
    options.add_options()
            ("h,help", "print usage")
            ("g,gpu", "number of GPU on which to run", cxxopts::value<int>()->default_value("0"))
            ("t,tbb", "use TBB with specified number of threads", cxxopts::value<int>()->default_value("0"))
            ("f,float", "run in single-precision")
            ("x,truncation", "enable truncation")
            ("i,iterations", "number of iterations", cxxopts::value<int>()->default_value("1"))
            ("r,row_locations", "number of row locations", cxxopts::value<int>()->default_value("4"))
            ("c,column_locations", "number of column locations", cxxopts::value<int>()->default_value("3"))
            ("d,dimension", "number of dimensions", cxxopts::value<int>()->default_value("2"))
            ("internal", "use internal dimension", cxxopts::value<bool>()->default_value("false"))
            ("missing", "allow for missing entries", cxxopts::value<bool>()->default_value("false"))
            ("sse", "use hand-rolled SSE", cxxopts::value<bool>()->default_value("false"))
            ("avx", "use hand-rolled AVX", cxxopts::value<bool>()->default_value("false"))
            ("avx512", "use hand-rolled AVX-512", cxxopts::value<bool>()->default_value("false"))
            ("show", "show results for each iteration")
            ;

    cxxopts::ParseResult result;

    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what()  << std::endl << std::endl;
        std::cout << options.help() << std::endl;
        exit(-1);
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

	long seed = 666L;

	auto prng = std::mt19937(seed);
	auto prng2 = std::mt19937(seed);

	std::cout << "Loading data" << std::endl;

    int embeddingDimension = result["dimension"].as<int>();
    int rowLocationCount = result["row_locations"].as<int>();
    int columnLocationCount = result["column_locations"].as<int>();

	bool updateAllLocations = true;

	long flags = 0L;

	auto normal = std::normal_distribution<double>(0.0, 1.0);
	auto uniform = std::uniform_int_distribution<int>(0, rowLocationCount + columnLocationCount - 1);
	auto binomial = std::bernoulli_distribution(0.75);
	auto normalData = std::normal_distribution<double>(0.0, 1.0);
	auto toss = std::bernoulli_distribution(0.25);

#ifdef USE_TBB
	std::shared_ptr<tbb::global_control> task{nullptr};
#endif

  int deviceNumber = -1;
  int threads = 0;
    if (result["gpu"].as<int>() > 0) {
		std::cout << "Running on GPU" << std::endl;
		flags |= mds::Flags::OPENCL;
        deviceNumber = result["gpu"].as<int>() - 1;
	} else {
		std::cout << "Running on CPU" << std::endl;
		threads = result["tbb"].as<int>();
		if (threads != 0) {
#ifdef USE_TBB		
			std::cout << "Using TBB with " << threads << " out of "
                << tbb::this_task_arena::max_concurrency()
			          << " threads" << std::endl;
			flags |= mds::Flags::TBB;
			task = std::make_shared<tbb::global_control>(tbb::global_control::max_allowed_parallelism, threads);
#endif
		}
	}

	if (result["float"].as<bool>()) {
		std::cout << "Running in single-precision" << std::endl;
		flags |= mds::Flags::FLOAT;
	} else {
		std::cout << "Running in double-precision" << std::endl;
	}

    int simdCount = 0;
    auto simd = "no simd";
    if (result["sse"].as<bool>()){
        ++simdCount;
        simd = "sse";
    }
    if (result["avx"].as<bool>()){
        ++simdCount;
        simd = "avx";
    }
    if (result["avx512"].as<bool>()){
        ++simdCount;
        simd = "avx512";
    }

    if (simdCount > 0) {
#if not defined(USE_SSE) && not defined(USE_AVX) && not defined(USE_AVX512)
        std::cerr << "SIMD is not implemented" << std::endl;
        exit(-1);
#else
        if (simdCount > 1) {
            std::cerr << "Can not request more than one SIMD simultaneously" << std::endl;
            exit(-1);
        }
        if (result["avx512"].as<bool>()) {
#ifndef USE_AVX512
            std::cerr << "AVX-512 is not implemented" << std::endl;
            exit(-1);
#else
            flags |= mds::Flags::AVX512;
#endif // USE_AVX512

		} else if (result["avx"].as<bool>()) {
#ifndef USE_AVX
			std::cerr << "AVX is not implemented" << std::endl;
			exit(-1);
#else
            flags |= mds::Flags::AVX;
#endif // USE_AVX
        } else {
            flags |= mds::Flags::SSE;
        }
#endif // not defined(USE_SSE) && not defined(USE_AVX) && not defined(USE_AVX512)
	}

	bool truncation = false;
	if (result["truncation"].as<bool>()) {
		std::cout << "Enabling truncation" << std::endl;
		flags |= mds::Flags::LEFT_TRUNCATION;
		truncation = true;
	}

	bool internalDimension = result["internal"].as<bool>();

	mds::Layout layout = mds::Layout(rowLocationCount, columnLocationCount);

	mds::SharedPtr instance = mds::factory(embeddingDimension, layout, flags, deviceNumber, threads);

	bool missing = result["missing"].as<bool>();
	if (missing) {
        std::cout << "Allowing for missingness" << std::endl;
	}

	auto elementCount = rowLocationCount * columnLocationCount;
	std::vector<double> data(elementCount);
	//int fixed = 1;
	for (int i = 0; i < rowLocationCount; ++i) {
	    for (int j = 0; j < columnLocationCount; ++j) {

	        const double draw = normalData(prng);
	        double distance = draw * draw;

	        if (missing && toss(prng2)) {
	            distance = NAN;
	        }
	        //distance = fixed++;
	        data[i * columnLocationCount + j] = distance;
	    }
	}

	instance->setPairwiseData(&data[0], elementCount);

	int dataDimension = internalDimension ? instance->getInternalDimension() : embeddingDimension;

	std::vector<double> location(dataDimension);
	std::vector<double> allLocations;
	if (updateAllLocations) {
		allLocations.resize(dataDimension * (rowLocationCount + columnLocationCount));
	}

	double total = 0.0;
	for (int i = 0; i < (rowLocationCount + columnLocationCount); ++i) {
		generateLocation(location, normal, prng);
		instance->updateLocations(i, &location[0], dataDimension);
	}

    int gradientIndex = 1;

	double precision = 1.0;
	instance->setParameters(&precision, 1);

	instance->makeDirty();
	auto logLik = instance->getSumOfIncrements();

    std::vector<double> gradient((rowLocationCount + columnLocationCount) * dataDimension, 0.0);

    instance->getLogLikelihoodGradient(gradient.data(),
                                       (rowLocationCount + columnLocationCount) *
                                         dataDimension);
    double sumGradient = gradient[gradientIndex];

	std::cout << "Starting MDS benchmark" << std::endl;
	auto startTime = std::chrono::steady_clock::now();

	int iterations = result["iterations"].as<int>();

	double timer = 0;
    double timer2 = 0;
    logLik = 0.0;

	for (auto itr = 0; itr < iterations; ++itr) {

// 	    double startDiagnostic = instance->getDiagnostic();

		instance->storeState();

		if (updateAllLocations) {
			generateLocation(allLocations, normal, prng);
			instance->updateLocations(-1, &allLocations[0], dataDimension * (rowLocationCount + columnLocationCount));
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

        if (result["show"].as<bool>()) {
            std::cout << "log-likelihood = " << inc << std::endl;
        }

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

        instance->getLogLikelihoodGradient(gradient.data(),
                                           (rowLocationCount + columnLocationCount) * dataDimension);

        auto duration2 = std::chrono::steady_clock::now() - startTime2;
        timer2 += std::chrono::duration<double, std::milli>(duration2).count();

        if (result["show"].as<bool>()) {
            std::cout << "gradient = [" << gradient[0];
            for (int j = 1; j < gradient.size(); ++j) {
                std::cout << ", " << gradient[j];
            }
            std::cout << "]" << std::endl;
        }

//        sumGradient += std::accumulate(std::begin(gradient), std::end(gradient), 0.0) + 1;
        sumGradient += gradient[gradientIndex];

	}
	logLik /= iterations;
    sumGradient /= iterations;
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
    outfile << deviceNumber << " " << threads << " " << simd << " " << rowLocationCount <<
      " " << columnLocationCount << " " << embeddingDimension << " " << iterations << " " << timer << " " << timer2 << "\n" ;
	outfile.close();

}
