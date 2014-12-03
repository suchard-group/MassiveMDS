#ifndef _ABSTRACTMULTIDIMENSIONALSCALING_HPP
#define _ABSTRACTMULTIDIMENSIONALSCALING_HPP

class AbstractMultiDimensionalScaling {
public:
    AbstractMultiDimensionalScaling(int embeddingDimension, int elementCount, long flags) 
        : embeddingDimension(embeddingDimension), elementCount(elementCount), flags(flags)
        { }
         
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
    int elementCount;
    long flags;    
};

template <typename Z>
class MultiDimensionalScaling : public AbstractMultiDimensionalScaling {
public:
    MultiDimensionalScaling(int embeddingDimension, int elementCount, long flags)
        : AbstractMultiDimensionalScaling(embeddingDimension, elementCount, flags),
          storage(embeddingDimension * embeddingDimension) { }
             
    virtual ~MultiDimensionalScaling() { }
            
    void updateLocations(int, double*, size_t) { }
    
    double calculateLogLikelihood() { return 0.0; }
    
    void storeState() { }
    
    void restoreState() { }
    
    void setPairwiseData(double*, size_t) { }
    
    void setParameters(double*, size_t) { }
    
    void makeDirty() { }
    
private:
    std::vector<Z> storage;    
    
};

#endif // _ABSTRACTMULTIDIMENSIONALSCALING_HPP