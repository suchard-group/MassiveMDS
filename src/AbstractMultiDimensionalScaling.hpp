#ifndef _ABSTRACTMULTIDIMENSIONALSCALING_HPP
#define _ABSTRACTMULTIDIMENSIONALSCALING_HPP

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <future>

//#include <xmmintrin.h>
#include <emmintrin.h>

#include "MemoryManagement.hpp"
#include "ThreadPool.h"

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

template <typename T>
struct DetermineType;

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
          isStoredSquaredResidualsEmpty(false) 
          , nThreads(4), pool(nThreads)
    { 
    
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
		// double delta = 0.0;
		
		const int i = updatedLocation;
		isStoredSquaredResidualsEmpty = false;
// 		std::copy(
// 			begin(squaredResiduals) + i * locationCount,
// 			begin(squaredResiduals) + (i + 1) * locationCount,
// 			begin(storedSquaredResiduals)
// 		);
		
#if 1		
		auto start  = begin(*locationsPtr) + i * embeddingDimension;
		auto offset = begin(*locationsPtr);
#else
		auto start  = &(*locationsPtr)[0] + i * embeddingDimension;
		auto offset = &(*locationsPtr)[0];
#endif
		
// 		DetermineType<decltype(start)>();

		double delta = 
		
//#define THREADS		
#ifdef THREADS
		accumulate_thread(0, locationCount, double(0), 
#else		
		accumulate(0, locationCount, double(0), 
#endif		
		
			[this, i, &offset, &start](const int j) {
                const auto distance = calculateDistance(
                    start,
                    offset,
                    embeddingDimension
                );
                offset += embeddingDimension;
                
                const auto residual = distance - observations[i * locationCount + j];                
                const auto squaredResidual = residual * residual;
            
            	// store old value
            	const auto oldSquaredResidual = squaredResiduals[i * locationCount + j];
            	storedSquaredResiduals[j] = oldSquaredResidual;
            	
                const auto inc = squaredResidual - oldSquaredResidual;            
            
            	// store new value
                squaredResiduals[i * locationCount + j] = squaredResidual;
                squaredResiduals[j * locationCount + i] = squaredResidual;                  
                
                return inc;
            }				
		);
				
// #if 1
// #if 1
//         for_each(0, locationCount, 
// #else
//         for_each_auto<2>(0, locationCount, 
// #endif
//             [this, i, &delta, &offset, &start](const int j) {
//                 const auto distance = calculateDistance(
//                     start,
//                     offset,
//                     embeddingDimension
//                 );
//                 offset += embeddingDimension;
//                 
//                 const auto residual = distance - observations[i * locationCount + j];                
//                 const auto squaredResidual = residual * residual;
//             
//                 delta += squaredResidual - squaredResiduals[i * locationCount + j];            
//             
//                 squaredResiduals[i * locationCount + j] = squaredResidual;
//                 squaredResiduals[j * locationCount + i] = squaredResidual;                  
//             }
//         );        
// #else		
// 		for (int j = 0; j < locationCount; ++j) {
// 			// TODO Code duplication with above
// 			const auto distance = calculateDistance(
// 				begin(*locationsPtr) + i * embeddingDimension,
// 				begin(*locationsPtr) + j * embeddingDimension,
// 				embeddingDimension
// 			);
// 			const auto residual = distance - observations[i * locationCount + j];
// 			const auto squaredResidual = residual * residual;
// 			
// 			delta += squaredResidual - squaredResiduals[i * locationCount + j];
// 			
// 			squaredResiduals[i * locationCount + j] = squaredResidual;
// 			squaredResiduals[j * locationCount + i] = squaredResidual;		
// 		}
// #endif
		
		sumOfSquaredResiduals += delta;
	}
	
#if 0	
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
#else
    
    
//#define VECTOR    
#ifdef VECTOR    
    double calculateDistance(double* x, double* y, int length) const {
//     std::cerr << "A";
        using Vec2 = __m128d;
        
        Vec2 vecX = _mm_load_pd(x);        
        Vec2 vecY = _mm_load_pd(y);
        vecX = _mm_sub_pd(vecX, vecY);
        vecX = _mm_mul_pd(vecX, vecX);
#if 1
        double r[2];
        _mm_store_pd(r,vecX);
        
        return std::sqrt(r[0] + r[1]);
#else
        double r;
        vecX =  _mm_hadd_pd(vecX, vecX);
        _mm_store_ps(r, vecX);
        return std::sqrt(r);
#endif        
    }
#else    
    double calculateDistance(double* x, double* y, int length) const {
//          std::cerr << "B";
        double r = 0.0;
        for (int i = 0; i < 2; ++i, ++x, ++y) {
            const auto difference = *x - *y;
            r += difference * difference;
        }
        return std::sqrt(r);
      
    }    
#endif // VECTOR
    template <typename Iterator>
    double calculateDistance(Iterator x, Iterator y, int length) const {
    
        auto sum = static_cast<double>(0);
        for (int i = 0; i < 2; ++i, ++x, ++y) {
            const auto difference = *x - *y;
            sum += difference * difference;
        }
        return std::sqrt(sum); 
    }
    
#endif	
	
#if 0

    template <size_t N> struct uint_{ };
 
    template <size_t N, typename Lambda, typename IterT>
    inline void unroller(const Lambda& f, const IterT& iter, uint_<N>) {
        unroller(f, iter, uint_<N-1>());
        f(iter + N);
    }
 
    template <typename Lambda, typename IterT>
    inline void unroller(const Lambda& f, const IterT& iter, uint_<0>) {
        f(iter);
    }

	template <typename Integer, typename Function>
	inline void for_each(Integer begin, Integer end, Function function) {
	    for (; begin != end; begin += 4) {
	        function(begin + 0);
	        function(begin + 1);
	        function(begin + 2);
	        function(begin + 3);	        
	    }
	}
	
	template <size_t UnrollFact, typename Integer, typename Function>
	inline void for_each_auto(Integer begin, Integer end, Function function) {
	    for (; begin != end; begin += UnrollFact) {
	        unroller(function, begin,  uint_<UnrollFact-1>());
	    }
	}	
#else	
	template <typename Integer, typename Function>
	inline void for_each(Integer begin, Integer end, Function function) {
	    for (; begin != end; ++begin) {
	        function(begin);
	    }
	}
	
	template <typename Integer, typename Function, typename Real>
	inline Real accumulate(Integer begin, Integer end, Real sum, Function function) {
		for (; begin != end; ++begin) {
			sum += function(begin);
		}
		return sum;
	}
	
	template <typename Integer, typename Function, typename Real>
	inline Real accumulate_thread(Integer begin, Integer end, Real sum, Function function) {
		std::vector<std::future<Real>> results;
		
		int chunkSize = (end - begin) / nThreads;
		int start = 0;
		
		for (int i = 0; i < nThreads - 1; ++i, start += chunkSize) {
			results.emplace_back(std::async([=] {
				return accumulate(begin + start, begin + start + chunkSize, 0.0, function);
			}));					
		}
		results.emplace_back(std::async([=] {
			return accumulate(begin + start, end, 0.0, function);
		}));
				
		for (auto&& result : results) {
			sum += result.get();
		}
		return sum;
	}	
	
	template <typename Integer, typename Function, typename Real>
	inline Real accumulate_omp(Integer begin, Integer end, Real sum, Function function) {
		#pragma omp
		for (; begin != end; ++begin) {
			sum += function(begin);
		}
		return sum;
	}		
	
	
	template <typename Integer, typename Function, typename Real>
	inline Real accumulate_thread_pool(Integer begin, Integer end, Real sum, Function function) {
		std::vector<std::future<Real>> results;
		
		int chunkSize = (end - begin) / nThreads;
		int start = 0;
		
		for (int i = 0; i < nThreads - 1; ++i, start += chunkSize) {
			results.emplace_back(
				pool.enqueue([=] {
					return accumulate(
						begin + start,
						begin + start + chunkSize,
						Real(0),
						function);					
				})			
			);							
		}
		results.emplace_back(
			pool.enqueue([=] {
				return accumulate(
					begin + start,
					end, 
					Real(0),
					function);
			})	
		
		); 
		
		Real total = static_cast<Real>(0);
		for (auto&& result : results) {
			total += result.get();
		}
		return total;
	}	
	
	
#endif
    
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
    
    int nThreads;
    ThreadPool pool;
        
};

} // namespace mds

#endif // _ABSTRACTMULTIDIMENSIONALSCALING_HPP