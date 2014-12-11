#ifndef _ABSTRACTMULTIDIMENSIONALSCALING_HPP
#define _ABSTRACTMULTIDIMENSIONALSCALING_HPP

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

#include "MemoryManagement.hpp"

namespace mds {

class AbstractMultiDimensionalScaling {
public:
    AbstractMultiDimensionalScaling(int embeddingDimension, int locationCount, long flags) 
        : embeddingDimension(embeddingDimension), locationCount(locationCount), 
          observationCount(locationCount * (locationCount - 1) / 2),
          flags(flags) { }
         
    virtual ~AbstractMultiDimensionalScaling() { }
    
    // Interface    
    virtual void updateLocations(int, double*, size_t) = 0;    
    virtual double calculateLogLikelihood() = 0;    
    virtual void storeState() = 0;    
    virtual void restoreState() = 0;      
    virtual void setPairwiseData(double*, size_t)  = 0;
    virtual void setParameters(double*, size_t) = 0;
    virtual void makeDirty() = 0;
    
protected:
    int embeddingDimension;
    int locationCount;    
    int observationCount;
    long flags; 
    
    int updatedLocation = -1;
    bool residualsKnown = false;
    bool sumOfSquaredResidualsKnown = false;
    bool isLeftTruncated = false;               
};

template <typename RealType>
class MultiDimensionalScaling : public AbstractMultiDimensionalScaling {
public:
    MultiDimensionalScaling(int embeddingDimension, int locationCount, long flags)
        : AbstractMultiDimensionalScaling(embeddingDimension, locationCount, flags),
//           precision(0.0), storedPrecision(0.0),
          sumOfSquaredResiduals(0.0), storedSumOfSquaredResiduals(0.0),
          observations(locationCount * locationCount),  
                  
          locations0(locationCount * embeddingDimension),
		  locations1(locationCount * embeddingDimension),
		  locationsPtr(&locations0),
		  storedLocationsPtr(&locations1),
		  
          squaredResiduals(locationCount * locationCount),
          storedSquaredResiduals(locationCount),          
          isStoredSquaredResidualsEmpty(false) { 
    
    	std::cout << "ctor MultiDimensionalScaling" << std::endl;    
    }
             
    virtual ~MultiDimensionalScaling() { }
            
    void updateLocations(int locationIndex, double* location, size_t length) {

    	assert(length == embeddingDimension);
     
    	if (updatedLocation != - 1) {
    		// more than one location updated -- do a full recomputation
    		residualsKnown = false;
    		//storedSquaredResidualsPtr = nullptr;
    		isStoredSquaredResidualsEmpty = true;
    		
    	}
    	    	
    	updatedLocation = locationIndex;
    	std::copy(location, location + length, 
    		begin(*locationsPtr) + locationIndex * embeddingDimension 
    		// TODO Check major-format
    	);
    	
    	sumOfSquaredResidualsKnown = false;    
    }
    
    double calculateLogLikelihood() { 
         
    	if (!sumOfSquaredResidualsKnown) {
    		if (!residualsKnown) {
    			computeSumOfSquaredResiduals();
    		} else {
    		
    			updateSumOfSquaredResiduals();
    		}
    		sumOfSquaredResidualsKnown = true;    	
    	}
    	    	
//     	double logLikelihood = 
//     			  (0.5 * std::log(precision) * observationCount) 
//     			- (0.5 * precision * sumOfSquaredResiduals);    			
//     	return logLikelihood;

		return sumOfSquaredResiduals;   	
 	}
    
    void storeState() {
    	storedSumOfSquaredResiduals = sumOfSquaredResiduals;    	
    	std::copy(begin(*locationsPtr), end(*locationsPtr), 
    		begin(*storedLocationsPtr));
    	
    	//storedSquaredResidualsPtr = nullptr;
    	isStoredSquaredResidualsEmpty = true;
//     	storedPrecision = precision;
    	
    	updatedLocation = -1;
    }
    
    void restoreState() { 
    	sumOfSquaredResiduals = storedSumOfSquaredResiduals;
    	sumOfSquaredResidualsKnown = true;
    	
//     	if (storedSquaredResidualsPtr != nullptr) {    	
		if (!isStoredSquaredResidualsEmpty) {
    		std::copy(
    			begin(storedSquaredResiduals),
    			end(storedSquaredResiduals),
    			begin(squaredResiduals) + updatedLocation * locationCount
    		);
    		for (int j = 0; j < locationCount; ++j) {
    			squaredResiduals[j * locationCount + updatedLocation] 
    				= storedSquaredResiduals[j];  		
    		}    	
    	}
    	
    	auto tmp1 = storedLocationsPtr;
    	storedLocationsPtr = locationsPtr;
    	locationsPtr = tmp1;
    	
//     	precision = storedPrecision;
    	    	
    	residualsKnown = true;    
    }
    
    void setPairwiseData(double* data, size_t length) {
		assert(length == observations.size()); 		
		std::copy(data, data + length, begin(observations));    
    }
    
    void setParameters(double* data, size_t length) { 
//     	assert(length == 1);
//     	precision = static_cast<RealType>(data[0]);		
		assert(length == 0); // Do not call
    }
    
    void makeDirty() {
    	sumOfSquaredResidualsKnown = false;
    	residualsKnown = false;
    }
    
	void computeSumOfSquaredResiduals() {
		sumOfSquaredResiduals = 0.0;
		for (int i = 0; i < locationCount; ++i) {
			for (int j = 0; j < locationCount; ++j) {
				const auto distance = calculateDistance(
					begin(*locationsPtr) + i * embeddingDimension,
					begin(*locationsPtr) + j * embeddingDimension,
					embeddingDimension
				);
				const auto residual = distance - observations[i * locationCount + j];
				const auto squaredResidual = residual * residual;
				squaredResiduals[i * locationCount + j] = squaredResidual;
				squaredResiduals[j * locationCount + i] = squaredResidual;
				sumOfSquaredResiduals += squaredResidual;
			}
		}	
	    
    	sumOfSquaredResiduals /= 2.0;
    
	    residualsKnown = true;
	    sumOfSquaredResidualsKnown = true;
	}
	
	void updateSumOfSquaredResiduals() {
		double delta = 0.0;
		
		const int i = updatedLocation;
		isStoredSquaredResidualsEmpty = false;
		std::copy(
			begin(squaredResiduals) + i * locationCount,
			begin(squaredResiduals) + (i + 1) * locationCount,
			begin(storedSquaredResiduals)
		);
		
		for (int j = 0; j < locationCount; ++j) {
			// TODO Code duplication with above
			const auto distance = calculateDistance(
				begin(*locationsPtr) + i * embeddingDimension,
				begin(*locationsPtr) + j * embeddingDimension,
				embeddingDimension
			);
			const auto residual = distance - observations[i * locationCount + j];
			const auto squaredResidual = residual * residual;
			
			delta += squaredResidual - squaredResiduals[i * locationCount + j];
			
			squaredResiduals[i * locationCount + j] = squaredResidual;
			squaredResiduals[j * locationCount + i] = squaredResidual;		
		}
		
		sumOfSquaredResiduals += delta;
	}
	
	template <typename Iterator>
	double calculateDistance(Iterator x, Iterator y, int length) const
		//-> decltype(Iterator::value_type) 
		{
	
		auto sum = //Iterator::value_type(0);
					static_cast<double>(0);
		for (int i = 0; i < length; ++i, ++x, ++y) {
			const auto difference = *x - *y;
			sum += difference * difference;
		}
		return std::sqrt(sum);
	}
    
private:
// 	double precision;
// 	double storedPrecision;
	
    double sumOfSquaredResiduals;
    double storedSumOfSquaredResiduals;		
		
    mm::MemoryManager<RealType> observations;
    
    mm::MemoryManager<RealType> locations0;
    mm::MemoryManager<RealType> locations1;
    
    mm::MemoryManager<RealType>* locationsPtr;
    mm::MemoryManager<RealType>* storedLocationsPtr;        
    
    mm::MemoryManager<RealType> squaredResiduals;   
    mm::MemoryManager<RealType> storedSquaredResiduals;    
    bool isStoredSquaredResidualsEmpty;
        
};

} // namespace mds

#endif // _ABSTRACTMULTIDIMENSIONALSCALING_HPP