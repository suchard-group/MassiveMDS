
MassiveMDS: massively parallel multidimensional scaling library
===

MassiveMDS facilitates fast Bayesian MDS through GPU, multi-core CPU, and SIMD vectorization powered implementations of the Hamiltonian Monte Carlo algorithm. 
The package may be built either as a standalone library or as an R package relying on Rcpp. 

# Standalone library

### Compilation

The standalone build requires CMake Version $\ge$ 2.8. Use the terminal to navigate to directory `mds/build`.

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


# R package

[![Build Status](https://travis-ci.com/suchard-group/MassiveMDS.svg?token=hAQxdsJP3XZzS5QwgS3M&branch=master)](https://travis-ci.com/suchard-group/MassiveMDS)

[![AppVeyor build status](https://ci.appveyor.com/api/projects/status/github/suchard-group/MassiveMDS?branch=master&svg=true)](https://ci.appveyor.com/project/suchard-group/MassiveMDS)

### Compilation

The R package requires Rcpp, RcppParallel, and RcppXsimd. First, open the MassiveMDS project located in `mds/src`. In R, install RcppXsimd with the following code.

```
devtools::install_github("OHDSI/RcppXsimd")
```

Then build the package.

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

