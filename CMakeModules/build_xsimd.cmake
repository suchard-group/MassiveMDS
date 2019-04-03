SET(VER 7.1.3)
SET(URL https://github.com/QuantStack/xsimd/archive/${VER}.tar.gz)
SET(MD5 1d831e7f6ca87aa75b09c6e88911c18f)

SET(thirdPartyDir "${CMAKE_BINARY_DIR}/third_party")
SET(srcDir "${thirdPartyDir}/xsimd-${VER}")
SET(archive ${srcDir}.tar.gz)
SET(inflated ${srcDir}-inflated)

# the config to be used in the code
SET(xsimd_INCLUDE_DIRS "${srcDir}/include")
file(COPY ${xsimd_INCLUDE_DIRS}/xsimd DESTINATION ${thirdPartyDir})

# do we have to do it again?
SET(doExtraction ON)
IF(EXISTS "${inflated}")
    FILE(READ "${inflated}" extractedMD5)
    IF("${extractedMD5}" STREQUAL "${MD5}")
        # nope, everything looks fine
        return()
    ENDIF()
ENDIF()

# lets get and extract xsimd

MESSAGE(STATUS "xsimd...")
IF(EXISTS "${archive}")
    FILE(MD5 "${archive}" md5)
    IF(NOT "${md5}" STREQUAL "${MD5}")
        FILE(REMOVE "${archive}")
        MESSAGE(FATAL_ERROR "  wrong check sum ${md5}")
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
    MESSAGE(FATAL_ERROR "${archive}: Invalid check sum ${md5}. Expected was ${MD5}")
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
