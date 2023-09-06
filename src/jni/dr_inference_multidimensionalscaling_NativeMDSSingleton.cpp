#include <memory>
#include <vector>
#include <iostream>

// #include "MultiDimensionalScaling.hpp"
// #include "AbstractMultiDimensionalScaling.hpp"
#include "NewMultiDimensionalScaling.hpp"
#include "dr_inference_multidimensionalscaling_NativeMDSSingleton.h"

typedef std::shared_ptr<mds::AbstractMultiDimensionalScaling> InstancePtr;
std::vector<InstancePtr> instances;
  
extern "C"
JNIEXPORT jint JNICALL Java_dr_inference_multidimensionalscaling_NativeMDSSingleton_initialize
  (JNIEnv *, jobject, jint embeddingDimension, jint elementCount1, jint elementCount2, jlong flags, jint device, jint threads) {
    mds::Layout layout = (elementCount1 == 0) ?
    	mds::Layout(elementCount1) :
    	mds::Layout(elementCount1, elementCount2);
    
    instances.emplace_back(
		mds::factory(embeddingDimension, layout, flags, device, threads)
    );
    return instances.size() - 1;
  }  

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_multidimensionalscaling_NativeMDSSingleton_updateLocations
  (JNIEnv *env, jobject, jint instance, jint index, jdoubleArray xArray) {
  	jsize len = env->GetArrayLength(xArray);
  	jdouble* x = env->GetDoubleArrayElements(xArray, NULL);

    instances[instance]->updateLocations(index, x, len);

    env->ReleaseDoubleArrayElements(xArray, x, JNI_ABORT);
}

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_inference_multidimensionalscaling_NativeMDSSingleton_getSumOfIncrements
  (JNIEnv *, jobject, jint instance) {
    return instances[instance]->getSumOfIncrements();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_multidimensionalscaling_NativeMDSSingleton_storeState
  (JNIEnv *, jobject, jint instance) {
    instances[instance]->storeState();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_multidimensionalscaling_NativeMDSSingleton_restoreState
  (JNIEnv *, jobject, jint instance) {
    instances[instance]->restoreState();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_multidimensionalscaling_NativeMDSSingleton_acceptState
  (JNIEnv *, jobject, jint instance) {
    instances[instance]->acceptState();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_multidimensionalscaling_NativeMDSSingleton_setPairwiseData
  (JNIEnv *env, jobject, jint instance, jdoubleArray xArray) {
  	jsize len = env->GetArrayLength(xArray);
  	jdouble* x = env->GetDoubleArrayElements(xArray, NULL);

    instances[instance]->setPairwiseData(x, len);

    env->ReleaseDoubleArrayElements(xArray, x, JNI_ABORT);
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_multidimensionalscaling_NativeMDSSingleton_getLocationGradient
  (JNIEnv *env, jobject, jint instance, jdoubleArray xArray) {
	jsize len = env->GetArrayLength(xArray);
	jdouble* x = env->GetDoubleArrayElements(xArray, NULL); // TODO: Try GetPrimitiveArrayCritical
	
	instances[instance]->getLogLikelihoodGradient(x, len);	
	
	env->ReleaseDoubleArrayElements(xArray, x, 0); // copy values back	  
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_multidimensionalscaling_NativeMDSSingleton_setParameters
  (JNIEnv *env, jobject, jint instance, jdoubleArray xArray) {
  	jsize len = env->GetArrayLength(xArray);
  	jdouble* x = env->GetDoubleArrayElements(xArray, NULL);

    instances[instance]->setParameters(x, len);

    env->ReleaseDoubleArrayElements(xArray, x, JNI_ABORT);
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_multidimensionalscaling_NativeMDSSingleton_makeDirty
  (JNIEnv *, jobject, jint instance) {
    instances[instance]->makeDirty();
}

extern "C"
JNIEXPORT jint JNICALL Java_dr_inference_multidimensionalscaling_NativeMDSSingleton_getInternalDimension
        (JNIEnv *, jobject, jint instance) {
    return instances[instance]->getInternalDimension();
}


// jsize len = (*env)->GetArrayLength(env, arr);
//     jdouble *partials = env->GetDoubleArrayElements(inPartials, NULL);
//
// 	jint errCode = (jint)beagleSetPartials(instance, bufferIndex, (double *)partials);
//
//     env->ReleaseDoubleArrayElements(inPartials, partials, JNI_ABORT);
//         // not using JNI_ABORT flag here because we want the values to be copied back...
//     env->ReleaseDoubleArrayElements(outPartials, partials, 0);
