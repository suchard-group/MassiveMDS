## Test environments
* local OS X install, R 3.5.3
* Ubuntu 14.04 (on travis-ci), R 3.6.0, R devel, gcc 7.0
* Windows Server 2012 R2 x64 (appveyor), R devel and release (mingw_32 and 64)
* win-builder (devel, release, oldrel)

## R CMD check results
* There were no ERRORs and WARNINGs.
* There are 3 NOTEs:
  - New submission
  - Possible misspellings in DESCRIPTION (of which, I believe, there are none)
  - CRAN Debian system complains
  
    "Compilation used the following non-portable flag(s):
    '-mavx' '-mavx2' '-mfma' '-msse4.2'"
    
    But this is after configure file checks for system availability.

## Downstream dependencies
There are currently no downstream dependencies.
