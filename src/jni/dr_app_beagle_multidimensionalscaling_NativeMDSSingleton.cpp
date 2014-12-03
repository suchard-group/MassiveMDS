#include <memory>
#include <vector>
#include <iostream>

#include "AbstractMultiDimensionalScaling.hpp"
#include "dr_app_beagle_multidimensionalscaling_NativeMDSSingleton.h"

typedef std::shared_ptr<AbstractMultiDimensionalScaling> InstancePtr;
std::vector<InstancePtr> instances;

extern "C"
JNIEXPORT jint JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_initialize
  (JNIEnv *, jobject, jint, jint, jlong flags) { 
    instances.emplace_back(std::make_shared<MultiDimensionalScaling<double>>());
    std::cerr << "flags: " << flags << std::endl;
    return instances.size() - 1; 
  }

extern "C"
JNIEXPORT void JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_updateLocations
  (JNIEnv *, jobject, jint, jint, jdoubleArray);

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_calculateLogLikelihood
  (JNIEnv *, jobject, jint) { return 0.0; }

extern "C"
JNIEXPORT void JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_storeState
  (JNIEnv *, jobject, jint) { }

extern "C"
JNIEXPORT void JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_restoreState
  (JNIEnv *, jobject, jint) { }

extern "C"
JNIEXPORT void JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_setPairwiseData
  (JNIEnv *, jobject, jint, jdoubleArray) { }

extern "C"
JNIEXPORT void JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_setParameters
  (JNIEnv *, jobject, jint, jdoubleArray) { }
