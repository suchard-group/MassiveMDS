
MassiveMDS: massively parallel multidimensional scaling library
===

MassiveMDS facilitates fast Bayesian MDS through GPU, multi-core CPU, and SIMD vectorization powered implementations of the Hamiltonian Monte Carlo algorithm. 
The package may be built either as a standalone library or as an R package relying on Rcpp.

GPU capabilities for either build require installation of OpenCL computing framework. See section **Configurations** below.

# R package

[![Build Status](https://travis-ci.com/suchard-group/MassiveMDS.svg?branch=master)](https://travis-ci.com/suchard-group/MassiveMDS)

[![Build status](https://ci.appveyor.com/api/projects/status/7cr6rmeqdwmo5unx?svg=true)](https://ci.appveyor.com/project/andrewjholbrook/massivemds)

### Compilation

Open the MassiveMDS project located in `mds/src` and build the package.

```
devtools::load_all(".")
```

### Testing

We use the `timeTest` function to time the different implementations of the MDS log likelihood and gradient calculations for `maxIts` iterations.  First check the serial implementation for 1000 locations.

```
timeTest(locationCount=1000)
```
We are interested in the number under `elapsed`.  The implementation using AVX SIMD should be roughly twice as fast.

```
timeTest(locationCount=1000, simd=2) 
```

Even faster should be a combination of AVX and a 4 core approach.

```
timeTest(locationCount=1000, simd=2, threads=4) 
```

The GPU implementation should be fastest of all.

```
timeTest(locationCount=1000, gpu=1) 
```

Not all GPUs have double precision capabilities. You might need to set `single=1` to use your GPU. If you have an eGPU connected, try fixing `gpu=2`. 

Speed computing the log likelihood and its gradient should translate directly to faster HMC times. Compare these implementations of HMC:

```
# generate high dimensional data
x       <- matrix(rnorm(5000),500,10)
data    <- as.matrix(dist(x))

hmc_1_0 <- hmcsampler(n_iter=100, data=data, learnPrec=FALSE, learnTraitPrec=FALSE)

hmc_3_2 <- hmcsampler(n_iter=100, data=data, learnPrec=FALSE, learnTraitPrec=FALSE, threads=3, simd=2)

hmc_gpu <- hmcsampler(n_iter=100, data=data, learnPrec=FALSE, learnTraitPrec=FALSE, gpu=1)

hmc_1_0$Time
hmc_3_2$Time
hmc_gpu$Time
```
Again, you might need to set `single=1` to get the GPU implementation working.  Hopefully, the elapsed times are fastest for the GPU implementation and slowest for the single threaded, no SIMD implementation.

### Example 1: user specified distance matrix

First, we randomly generate a distance matrix and use HMC to learn the Bayesian MDS posterior.  We use AVX and 4 CPU cores.  We also choose to learn the MDS likelihood precision (`learnPrec=TRUE`) and the `P` dimensional latent precision matrix (`learnTraitPrec=TRUE`). 

```
# generate high dimensional data
x       <- matrix(rnorm(5000),500,10)
data    <- as.matrix(dist(x))

# run HMC
hmc <- hmcsampler(n_iter=100, burnIn=0, data=data, learnPrec=TRUE, learnTraitPrec=TRUE, threads=3, simd=2, treeCov=FALSE, trajectory=0.1)
```
Look at trace plot of MDS precision.
```
plot(hmc$precision, type="l")
```

Hmmm, needs more samples. Let's speed it up with the GPU.

```
hmc <- hmcsampler(n_iter=200, burnIn=0, data=data, learnPrec=TRUE, learnTraitPrec=TRUE, gpu=1, single=1,  treeCov=FALSE, trajectory=0.1)

plot(hmc$target, type="l") # plot the negative log likelihood

```

Still not enough samples. Run the chain for 2000 iterations.

### Example 2: phylogenetic inference

```
beast <- readbeast()
hmc <- hmcsampler(n_iter=1000, burnIn=500, beast=beast, learnPrec=TRUE, learnTraitPrec=TRUE, threads=3, simd=2, treeCov=FALSE, trajectory=0.01)
```


# Standalone library

### Compilation

The standalone build requires CMake Version ≥ 2.8. Use the terminal to navigate to directory `mds/build`.

```
cd build
cmake ..
make
```

### Testing

Once the library is built, test the various implementation settings. The `benchmark` program computes the MDS log likelihood and its gradient for a given number of iterations and returns the time taken for each. First check the serial implementation for 1000 locations.

```
./benchmark --truncation --locations 1000 
```

The following implementation using AVX SIMD should be roughly twice as fast.

```
./benchmark --truncation --locations 1000 --avx
```

Even faster should be a combination of AVX and a 4 core approach.

```
./benchmark --truncation --locations 1000 --avx --tbb 4
```

The GPU implementation should be fastest of all. Make sure that your GPU can handle double precision floating points.  If not, make sure to toggle `--float`.  

```
./benchmark --truncation --locations 1000 --gpu 2
```

Test the different methods by increasing `iterations` and `locations`.



# Configurations

### OpenCL

Both builds of MassiveMDS rely on the OpenCL framework for their GPU capabilities. Builds using OpenCL generally require access to the OpenCL headers <https://github.com/KhronosGroup/OpenCL-Headers> and the shared library `OpenCL.so` (or dynamically linked library `OpenCL.dll` for Windows).  Since we have included the headers in the package, one only needs acquire the shared library. Vendor specific drivers include the OpenCL shared library and are available here:

NVIDIA <https://www.nvidia.com/Download/index.aspx?lang=en-us>

AMD <https://www.amd.com/en/support>

Intel <https://downloadcenter.intel.com/product/80939/Graphics-Drivers> .


Another approach is to download vendor specific SDKs, which also include the shared libraries. <https://github.com/cdeterman/gpuR/wiki/Installing-OpenCL> has more details on this approach.

#### OpenCL on Windows
Building the MassiveMDS R package on Windows with OpenCL requires copying (once installed) `OpenCl.dll` to the MassiveMDS library.  For a 64 bit machine use

```
cd MassiveMDS
scp /C/Windows/System32/OpenCL.dll inst/lib/x64
```
and for a 32 bit machine use the following.
```
cd MassiveMDS
scp /C/Windows/SysWOW64/OpenCL.dll inst/lib/i386
```
Finally, uncomment the indicated lines in `src/Makevars.win`.


### C++14 on Windows

Compiling with C++14 and the default Rtools Mingw64 compiler causes an error. Circumvent the error by running the following R code prior to build (cf. [RStan for Windows](https://github.com/stan-dev/rstan/wiki/Installing-RStan-from-source-on-Windows#configuration)).

```
dotR <- file.path(Sys.getenv("HOME"), ".R")
if (!file.exists(dotR))
  dir.create(dotR)
M <- file.path(dotR, "Makevars.win")
if (!file.exists(M))
  file.create(M) 
cat("\nCXX14FLAGS=-O3",
  "CXX14 = $(BINPREF)g++ -m$(WIN) -std=c++1y",
  "CXX11FLAGS=-O3", file = M, sep = "\n", append = TRUE)
```

### eGPU

External GPUs can be difficult to set up. For OSX, we have had success with `macOS-eGPU.sh`

<https://github.com/learex/macOS-eGPU> .

For Windows 10 on Mac (using bootcamp), we have had success with the `automate-eGPU EFI`

<https://egpu.io/forums/mac-setup/automate-egpu-efi-egpu-boot-manager-for-macos-and-windows/> .

The online community <https://egpu.io/> is helpful for other builds.
