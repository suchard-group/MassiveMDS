# Process this file with autoconf to produce a configure script.
AC_INIT(mds, 0.1, msuchard@ucla.edu)

# find R home and set CC/CFLAGS
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi

# platform check
RHOST=`"${R_HOME}/bin/Rscript" -e 'cat(Sys.info()[["sysname"]])'`

# get RcppParallel directory
RCPPPAR_LIB=`"${R_HOME}/bin/Rscript" -e "RcppParallel::RcppParallelLibs()"`

# Check for which host we are on and setup a few things
# specifically based on the host

OPENCL_FLAGS=
OPENCL_LIB=
SIMD_FLAGS="-DUSE_SIMD -DUSE_SSE -msse4.2" # -DUSE_AVX -mavx -mavx2 -mfma"

case $RHOST in
  Darwin* )
        # Do something specific for Mac OS X
        AC_CHECK_FILE(/System/Library/Frameworks/OpenCL.framework,
        	[	OPENCL_LIB="-framework OpenCL";
        		OPENCL_FLAGS="-DHAVE_OPENCL"	],
        	[])
        ;;
  Linux*)
        # Do something specific for linux
        AC_CHECK_LIB(OpenCL, clFinish,
        	[	OPENCL_FLAGS="-DHAVE_OPENCL";
        	 	OPENCL_LIB="-lOpenCL"	],
        	[])
        ;;
   *BSD*)
        # Do something specific for BSD
        AC_MSG_ERROR([BSD is not currently supported])
        ;;
    *)
        #Default Case
        AC_MSG_ERROR([Your platform is not currently supported])
        ;;
esac

AC_SUBST(OPENCL_LIB)
AC_SUBST(OPENCL_FLAGS)
AC_SUBST(SIMD_FLAGS)
AC_SUBST(RCPPPAR_LIB)

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
