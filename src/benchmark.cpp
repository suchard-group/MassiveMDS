
#include <chrono>
#include <random>
#include <iostream>

#include "AbstractMultiDimensionalScaling.hpp"

template <typename T, typename PRNG, typename D>
void generateLocation(T& locations, D& d, PRNG& prng) {
	for (auto& location : locations) {
		location = d(prng);
	}
}

int main(int argc, char* argv[]) {

	long seed = 666L;
		
	auto prng = std::mt19937(seed);

	std::cout << "Starting MDS benchmark" << std::endl;	
	auto startTime = std::chrono::steady_clock::now();
	
	int embeddingDimension = 2;
	int locationCount = 1000; 
	long flags = 0L;
	
	auto normal = std::normal_distribution<double>(0.0, 1.0);
	auto uniform = std::uniform_int_distribution<int>(0, locationCount - 1);
	auto binomial = std::bernoulli_distribution(0.75);	
	auto normalData = std::normal_distribution<double>(0.0, 1.0);	
	
	mds::MultiDimensionalScaling<double> instance{embeddingDimension, locationCount, flags};
	
	auto elementCount = locationCount * locationCount;
	std::vector<double> data(elementCount);
	for (auto& datum : data) {
	    double draw = normalData(prng);
		datum = draw * draw;
	}
	instance.setPairwiseData(&data[0], elementCount);
	
	std::vector<double> location(embeddingDimension);
	for (int i = 0; i < locationCount; ++i) {
		generateLocation(location, normal, prng);
		instance.updateLocations(i, &location[0], embeddingDimension);
	}
	
	instance.makeDirty();
	auto logLik = instance.calculateLogLikelihood();
	
	int iterations = 1000 * 100;
	
	for (auto itr = 0; itr < iterations; ++itr) {
		instance.storeState();
		
		int dimension = uniform(prng);
		generateLocation(location, normal, prng);
		instance.updateLocations(dimension, &location[0], embeddingDimension);
		double inc = instance.calculateLogLikelihood();
		logLik += inc;
		
		bool restore = binomial(prng);
		if (restore) {
			instance.restoreState();
		}
// 		std::cout << inc << std::endl;											
	}
	logLik /= iterations + 1;	
	
	auto endTime = std::chrono::steady_clock::now();
	auto duration = endTime - startTime;	
	std::cout << "AvgLogLik = " << logLik << std::endl;
	std::cout << "End MDS benchmark" << std::endl;
	std::cout << std::chrono::duration<double, std::milli> (duration).count() << " ms " 
			  << std::endl;
	
}