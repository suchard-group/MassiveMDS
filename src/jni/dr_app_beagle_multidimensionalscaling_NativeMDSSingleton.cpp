#include <memory>
#include <vector>
#include <iostream>

#include "AbstractMultiDimensionalScaling.hpp"
#include "dr_app_beagle_multidimensionalscaling_NativeMDSSingleton.h"

typedef std::shared_ptr<mds::AbstractMultiDimensionalScaling> InstancePtr;
std::vector<InstancePtr> instances;

extern "C"
JNIEXPORT jint JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_initialize
  (JNIEnv *, jobject, jint embeddingDimension, jint elementCount, jlong flags) { 
    instances.emplace_back(
        std::make_shared<mds::MultiDimensionalScaling<double>>(embeddingDimension, elementCount, flags));
    return instances.size() - 1; 
  }

extern "C"
JNIEXPORT void JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_updateLocations
  (JNIEnv *, jobject, jint instance, jint index, jdoubleArray x) {
    instances[instance]->updateLocations(index, nullptr, 0);  
}

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_calculateLogLikelihood
  (JNIEnv *, jobject, jint instance) {
    return instances[instance]->calculateLogLikelihood();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_storeState
  (JNIEnv *, jobject, jint instance) {
    instances[instance]->storeState();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_restoreState
  (JNIEnv *, jobject, jint instance) {
    instances[instance]->restoreState();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_setPairwiseData
  (JNIEnv *, jobject, jint instance, jdoubleArray x) {
    instances[instance]->setPairwiseData(nullptr, 0);
}

extern "C"
JNIEXPORT void JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_setParameters
  (JNIEnv *, jobject, jint instance, jdoubleArray x) {
    instances[instance]->setParameters(nullptr, 0);
}

extern "C"  
JNIEXPORT void JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_makeDirty
  (JNIEnv *, jobject, jint instance) {
    instances[instance]->makeDirty();
}
