---
output:
  html_document: default
  pdf_document: default
---
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

### Build status

[![Build Status](https://travis-ci.com/suchard-group/mds.svg?token=hAQxdsJP3XZzS5QwgS3M&branch=master)](https://travis-ci.com/suchard-group/mds)


# R package

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

