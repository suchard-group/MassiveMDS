#ifndef _ABSTRACTMULTIDIMENSIONALSCALING_HPP
#define _ABSTRACTMULTIDIMENSIONALSCALING_HPP

class AbstractMultiDimensionalScaling {
public:
    virtual ~AbstractMultiDimensionalScaling() { }
    virtual void initialize(int, int, long) = 0;
    virtual void pdateLocations(int, double*) = 0;
    virtual double calculateLogLikelihood() = 0;
    virtual void storeState() = 0;
    virtual void restoreState() = 0;
    virtual void setPairwiseData(int, double*)  = 0;
};

template <typename T>
class MultiDimensionalScaling : public AbstractMultiDimensionalScaling {
public:
    MultiDimensionalScaling() { }
    virtual ~MultiDimensionalScaling() { }
    void initialize(int, int, long) { }
    void pdateLocations(int, double*) { }
    double calculateLogLikelihood() { return 0.0; }
    void storeState() { }
    void restoreState() { }
    void setPairwiseData(int, double*) { }
};

#endif // _ABSTRACTMULTIDIMENSIONALSCALING_HPP