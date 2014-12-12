cmake_minimum_required(VERSION 2.6)
project(mds)

#set(CMAKE_CXX_COMPILER "/usr/local/bin/clang++-3.5")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -pedantic -msse4.2 -std=c++0x")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -pedantic -fPIC -std=c++0x")

# -mllvm -debug-pass=loop-vectorize  

find_package(JNI)

if (NOT JNI_FOUND)
	find_path (JNI_HEADER
		jni.h
		PATHS /usr/lib/gcc/x86_64-redhat-linux/4.4.7/ /u/local/apps/java/current/
		PATH_SUFFIXES include
	)
	message (STATUS "${JNI_HEADER}")
	set (JNI_INCLUDE_DIRS "${JNI_HEADER}")
	message (STATUS "${JNI_INCLUDE_DIRS}")

	if (JNI_HEADER)
		set (JNI_FOUND 1)
	endif(JNI_HEADER)
endif (NOT JNI_FOUND)


if (JNI_FOUND)
   message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
   message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")

include_directories(. ${CMAKE_SOURCE_DIR}/src ${JNI_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/jni)

set(JNI_SOURCE_FILES
	src/jni/dr_app_beagle_multidimensionalscaling_NativeMDSSingleton.cpp
	)
	
add_library(mds_jni SHARED ${JNI_SOURCE_FILES})

#IF(APPLE)
#  SET_TARGET_PROPERTIES(${SWIG_MODULE_<module_name>_REAL_NAME} PROPERTIES
#    PREFIX "lib" SUFFIX ".jnilib")
#ELSEIF(WIN32)
#  SET_TARGET_PROPERTIES(${SWIG_MODULE_<module_name>_REAL_NAME} PROPERTIES
#    PREFIX "" SUFFIX ".dll")
#ELSEIF(UNIX)
#  SET_TARGET_PROPERTIES(${SWIG_MODULE_<module_name>_REAL_NAME} PROPERTIES
#    PREFIX "lib" SUFFIX ".so")
#ENDIF(APPLE)

#add_definitions(-DDOUBLE_PRECISION)
#target_link_libraries(bsccs_jni base_bsccs-dp)

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark mds_jni)




else(JNI_FOUND)
	message (WARNING " No JNI found!")
endif(JNI_FOUND)
