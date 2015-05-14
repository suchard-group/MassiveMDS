SET(VER 1.3.3)
SET(URL https://github.com/ddemidov/vexcl/archive/${VER}.tar.gz)
SET(MD5 bab6d4cc5fa8886ef23fb8548b81429a)

SET(thirdPartyDir "${CMAKE_BINARY_DIR}/third_party")
SET(srcDir "${thirdPartyDir}/vexcl-${VER}")
SET(archive ${srcDir}.tar.gz)
SET(inflated ${srcDir}-inflated)

# the config to be used in the code
SET(VexCL_INCLUDE_DIRS "${srcDir}/vexcl")

# do we have to do it again?
SET(doExtraction ON)
IF(EXISTS "${inflated}")
    FILE(READ "${inflated}" extractedMD5)
    IF("${extractedMD5}" STREQUAL "${MD5}")
        # nope, everything looks fine
        return()
    ENDIF()
ENDIF()

# lets get and extract boost compute

MESSAGE(STATUS "VexXCL...")
IF(EXISTS "${archive}")
    FILE(MD5 "${archive}" md5)
    IF(NOT "${md5}" STREQUAL "${MD5}")
        MESSAGE("  wrong check sum ${md5}, redownloading")
        FILE(REMOVE "${archive}")
    ENDIF()
ENDIF()

IF(NOT EXISTS "${archive}")
    MESSAGE(STATUS "  getting ${URL}")
    FILE(DOWNLOAD "${URL}" ${archive}
        STATUS rv
        SHOW_PROGRESS)
ENDIF()

MESSAGE(STATUS "  validating ${archive}")
FILE(MD5 "${archive}" md5)
IF(NOT "${md5}" STREQUAL "${MD5}")
    MESSAGE(WARNING "${archive}: Invalid check sum ${md5}. Expected was ${MD5}")
    IF("${md5}" STREQUAL "d41d8cd98f00b204e9800998ecf8427e")
        MESSAGE(STATUS "Trying wget ${URL}")
        EXECUTE_PROCESS(COMMAND wget --no-check-certificate -O ${archive} ${URL})
        FILE(MD5 "${archive}" md5_)
        IF(NOT "${md5_}" STREQUAL "${MD5}")
            MESSAGE(FATAL_ERROR "${archive}: Invalid check sum ${md5_}. Expected was ${MD5}")
        ENDIF(NOT "${md5_}" STREQUAL "${MD5}")
        MESSAGE(STATUS "wget successful")
    ENDIF("${md5}" STREQUAL "d41d8cd98f00b204e9800998ecf8427e")
ENDIF()

IF(IS_DIRECTORY ${srcDir})
    MESSAGE(STATUS "  cleaning ${cleaning}")
    FILE(REMOVE_RECURSE ${srcDir})
ENDIF()

MESSAGE(STATUS "  extracting ${archive}")
FILE(MAKE_DIRECTORY ${srcDir})
EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E tar xfz ${archive}
    WORKING_DIRECTORY ${thirdPartyDir}
    RESULT_VARIABLE rv)
IF(NOT rv EQUAL 0)
    MESSAGE(FATAL_ERROR "'${archive}' extraction failed")
ENDIF()

FILE(WRITE ${inflated} "${MD5}")
