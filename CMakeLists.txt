CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
PROJECT(mds2)

OPTION(BUILD_NOSIMD "Build MDS without SIMD" OFF)
OPTION(BUILD_AVX512 "Build MDS with AVX-512" OFF)
OPTION(BUILD_AVX "Build MDS with AVX" OFF)
OPTION(BUILD_CUDA "Build MDS with a CUDA backend" OFF)
OPTION(BUILD_OPENCL "Build MDS with a OpenCL backend" OFF)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -pedantic -std=c++14 -Wno-deprecated")
else(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -g -pedantic -Wno-ignored-attributes -std=c++14")
endif(APPLE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -pedantic -fPIC -std=c++11")

# -Rpass=loop-vectorize

# -mllvm -debug-pass=loop-vectorize

if(APPLE)
   if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
        execute_process(COMMAND sysctl -q hw.optional.arm64
            OUTPUT_VARIABLE _sysctl_stdout
            ERROR_VARIABLE _sysctl_stderr
            RESULT_VARIABLE _sysctl_result
            )
        if(_sysctl_result EQUAL 0 AND _sysctl_stdout MATCHES "hw.optional.arm64:
 1")
            set(CMAKE_HOST_SYSTEM_PROCESSOR "arm64")
        endif()
    endif()
    set(MDS_OSX_NATIVE_ARCHITECTURE ${CMAKE_HOST_SYSTEM_PROCESSOR})

    if(MDS_OSX_NATIVE_ARCHITECTURE STREQUAL "arm64")
    	set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
        message(STATUS "macOS universal (x86_64 / arm64) build")
		# set(CMAKE_OSX_ARCHITECTURES "x86_64")
        # message(STATUS "macOS x86_64 build")        
    else()
        message(STATUS "macOS x86_64 build")
    endif()
endif(APPLE)

OPTION(USE_TBB "Use TBB" ON) # TODO does not work (need to comment out in factory.cpp)
OPTION(USE_SYSTEM_TBB "Use system TBB" ON)

if(USE_TBB)
  ADD_DEFINITIONS(-DUSE_TBB)

IF(USE_SYSTEM_TBB)
  FIND_PACKAGE(TBB REQUIRED)
ELSE()
  MESSAGE(STATUS "Compiling local TBB")
  if (NO_RTM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mno-rtm")
  endif()

  SET(TBB_BUILD_STATIC OFF CACHE BOOL         "Build TBB static library")
  SET(TBB_BUILD_TBBMALLOC OFF CACHE BOOL      "Build TBB malloc library")
  SET(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL "Build TBB malloc proxy library")
  SET(TBB_BUILD_TESTS OFF CACHE BOOL           "Build TBB tests and enable testing infrastructure")

  #ADD_SUBDIRECTORY(external/tbb)
  #SET(TBB_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/external/tbb/include")
  SET(TBB_LIBRARIES tbb)
  INCLUDE("${CMAKE_MODULE_PATH}/build_tbb.cmake")
ENDIF()
ENDIF(USE_TBB)

INCLUDE_DIRECTORIES(${TBB_INCLUDE_DIRS})
MESSAGE(STATUS "TBB directory: ${TBB_INCLUDE_DIRS}")

option(USE_SYSTEM_XSIMD "Use system XSIMD" OFF)
if(USE_SYSTEM_XSIMD)
  find_package(xsimd REQUIRED)
else()
  include("${CMAKE_MODULE_PATH}/build_xsimd.cmake")
endif()
message(STATUS "xsimd directory: ${xsimd_INCLUDE_DIRS}")


# OS Definitions
IF(UNIX)
  IF(APPLE)   #OSX
    ADD_DEFINITIONS(-DOS_MAC)

    SET(CMAKE_MACOSX_RPATH ON)
    SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
    SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    IF("${isSystemDir}" STREQUAL "-1")
      SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    ENDIF("${isSystemDir}" STREQUAL "-1")
  ELSE(APPLE) #Linux
    ADD_DEFINITIONS(-DOS_LNX)
  ENDIF()
ELSE(${UNIX}) #Windows
  ADD_DEFINITIONS(-DOS_WIN -DNOMINMAX)
ENDIF()

# Version information
INCLUDE(${CMAKE_MODULE_PATH}/Version.cmake)

IF(${BUILD_CUDA})
    ADD_SUBDIRECTORY(src/backend/cuda)
ENDIF()

IF(${BUILD_OPENCL})
    ADD_SUBDIRECTORY(src/backend/opencl)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_OPENCL")
ENDIF()

ADD_EXECUTABLE(bin2cpp ${CMAKE_MODULE_PATH}/bin2cpp.cpp)

MESSAGE(STATUS "TBB libraries: ${TBB_LIBRARIES}")

find_package(JNI)

#find_package(FindOpenCL)

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

   ##
   # Build boost program_options for benchmark
   ##
   INCLUDE("${CMAKE_MODULE_PATH}/build_boost_headers.cmake")
   file(GLOB BOOST_PO_FILES
           "${BoostHeaders_INCLUDE_DIRS}/libs/program_options/src/*.cpp"
           )
   include_directories(${BoostHeaders_INCLUDE_DIRS})
   include_directories(${xsimd_INCLUDE_DIRS})
   add_library(boost_program_options ${BOOST_PO_FILES})

   IF(${BUILD_NOSIMD})
       ADD_SUBDIRECTORY(src/backend/nosimd)
   ENDIF()

       ## Build MDS with SIMD
include_directories(. ${CMAKE_SOURCE_DIR}/src ${JNI_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/jni
	TBB_INCLUDE_DIRS
	)

set(JNI_SOURCE_FILES
	src/jni/dr_inference_multidimensionalscaling_NativeMDSSingleton.cpp
	src/factory.cpp
        src/MemoryManagement.hpp
        src/AbstractMultiDimensionalScaling.hpp
        src/NewMultiDimensionalScaling.hpp
        src/CDF.h
        src/Distance.hpp
	)


   set(AVX512_FLAGS "-DUSE_SIMD -DUSE_SSE -DUSE_AVX -DUSE_AVX512 -msse4.2 -mavx -mavx2 -mfma  -msse4.2 -mavx -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512ifma -mavx512vbmi")
   set(AVX_FLAGS "-DUSE_SIMD -DUSE_SSE -DUSE_AVX -msse4.2 -mfma  -msse4.2 -mavx")
   set(SSE_FLAGS "-DUSE_SIMD -DUSE_SSE -msse4.2")

   if(${BUILD_AVX512})
       set(SIMD_FLAGS "${AVX512_FLAGS}")
   else(${BUILD_AVX512})
       if(${BUILD_AVX})
           set(SIMD_FLAGS "${AVX_FLAGS}")
       else(${BUILD_AVX})
           set(SIMD_FLAGS "${SSE_FLAGS}")
       endif(${BUILD_AVX})
   endif(${BUILD_AVX512})

add_library(mds2_jni SHARED ${JNI_SOURCE_FILES})
set_target_properties(mds2_jni PROPERTIES COMPILE_FLAGS "${SIMD_FLAGS}")

if(BUILD_OPENCL)
	target_link_libraries(mds2_jni mds2_opencl)
endif(BUILD_OPENCL)

target_link_libraries(mds2_jni ${TBB_LIBRARIES})

##
# Build benchmark
##

# add_executable(benchmark src/benchmark.cpp)
# target_link_libraries(benchmark mds_jni)
# set_target_properties(benchmark PROPERTIES COMPILE_FLAGS "${SIMD_FLAGS}")
# target_link_libraries(benchmark ${TBB_LIBRARIES})
# target_link_libraries(benchmark boost_program_options)

add_executable(benchmark2 src/benchmark2.cpp)
target_link_libraries(benchmark2 mds2_jni)
set_target_properties(benchmark2 PROPERTIES COMPILE_FLAGS "${SIMD_FLAGS}")
target_link_libraries(benchmark2 ${TBB_LIBRARIES})
target_link_libraries(benchmark2 boost_program_options)

# add_executable(benchmark-san src/benchmark.cpp)
# set_target_properties(benchmark-san PROPERTIES COMPILE_FLAGS "-fsanitize=address")
# set_target_properties(benchmark-san PROPERTIES LINK_FLAGS "-fsanitize=address")
# set_target_properties(benchmark-san PROPERTIES COMPILE_FLAGS "${SSE_FLAGS}")
# target_link_libraries(benchmark-san mds_jni)
# target_link_libraries(benchmark-san ${TBB_LIBRARIES})
# target_link_libraries(benchmark-san boost_program_options)

add_executable(testme src/test.cpp)

##
# Packaging
##
include(${CMAKE_CURRENT_SOURCE_DIR}/CPack.txt)


else(JNI_FOUND)
	message (WARNING " No JNI found!")
endif(JNI_FOUND)
