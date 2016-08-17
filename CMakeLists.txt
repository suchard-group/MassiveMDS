CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(mds)

#set(CMAKE_CXX_COMPILER "/opt/clang6.0/bin/clang++")

OPTION(BUILD_CPU "Build MDS with a CPU backend" OFF)
OPTION(BUILD_CUDA "Build MDS with a CUDA backend" OFF)
OPTION(BUILD_OPENCL "Build MDS with a OpenCL backend" ON)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -pedantic -msse4.2 -std=c++0x -ffast-math -Rpass=loop-vectorize -Rpass-missed=loop-vectorize  -Rpass-analysis=loop-vectorize")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -pedantic -msse4.2 -std=c++0x -ffast-math -Rpass=loop-vectorize")

if(APPLE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -pedantic -msse4.2 -std=c++11")
else(APPLE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -g -pedantic -msse4.2 -std=c++11")
endif(APPLE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -pedantic -fPIC -std=c++11")

# -Rpass=loop-vectorize

# -mllvm -debug-pass=loop-vectorize

OPTION(USE_SYSTEM_TBB "Use system TBB" ON)
IF(USE_SYSTEM_TBB)
  FIND_PACKAGE(TBB REQUIRED)
ELSE()
  MESSAGE(STATUS "Compiling local TBB")
  ADD_SUBDIRECTORY(external/tbb)
  SET(TBB_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/external/tbb/include")  
  SET(TBB_LIBRARIES tbb)
  #INCLUDE("${CMAKE_MODULE_PATH}/build_tbb.cmake")  
ENDIF()
INCLUDE_DIRECTORIES(${TBB_INCLUDE_DIRS})
MESSAGE(STATUS "TBB directory: ${TBB_INCLUDE_DIRS}")


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

IF(${BUILD_CPU})
  ADD_SUBDIRECTORY(src/backend/cpu)
ENDIF()

IF(${BUILD_CUDA})
    ADD_SUBDIRECTORY(src/backend/cuda)
ENDIF()

IF(${BUILD_OPENCL})
    ADD_SUBDIRECTORY(src/backend/opencl)
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

include_directories(. ${CMAKE_SOURCE_DIR}/src ${JNI_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/jni
	TBB_INCLUDE_DIRS
	)

set(JNI_SOURCE_FILES
	src/jni/dr_inference_multidimensionalscaling_NativeMDSSingleton.cpp
	src/factory.cpp
	)

add_library(mds_jni SHARED ${JNI_SOURCE_FILES})
target_link_libraries(mds_jni mds_opencl)
target_link_libraries(mds_jni ${TBB_LIBRARIES})


##
# Build boost program_options for benchmark
##
INCLUDE("${CMAKE_MODULE_PATH}/build_boost_headers.cmake")
file(GLOB BOOST_PO_FILES
    "${BoostHeaders_INCLUDE_DIRS}/libs/program_options/src/*.cpp"
)
include_directories(${BoostHeaders_INCLUDE_DIRS})
add_library(boost_program_options ${BOOST_PO_FILES})

##
# Build benchmark
##

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark mds_jni)
target_link_libraries(benchmark ${TBB_LIBRARIES})
target_link_libraries(benchmark boost_program_options)

##
# Packaging
##
include(${CMAKE_CURRENT_SOURCE_DIR}/CPack.txt)


else(JNI_FOUND)
	message (WARNING " No JNI found!")
endif(JNI_FOUND)
