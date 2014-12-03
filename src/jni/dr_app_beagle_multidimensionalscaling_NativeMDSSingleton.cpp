#include "dr_app_beagle_multidimensionalscaling_NativeMDSSingleton.h"

extern "C"
JNIEXPORT void JNICALL Java_dr_app_beagle_multidimensionalscaling_NativeMDSSingleton_initialize
  (JNIEnv *, jobject, jint, jint, jint) { }

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
