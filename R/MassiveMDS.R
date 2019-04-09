#' MassiveMDS
#'
#' MassiveMDS facilitates fast Bayesian MDS through GPU, multi-core CPU, and SIMD vectorization powered implementations of the Hamiltonian Monte Carlo algorithm.
#' The package may be built either as a standalone library or as an R package relying on Rcpp.
#'
#' @docType package
#' @name MassiveMDS
#' @author Marc Suchard and Andrew Holbrook
#' @importFrom Rcpp evalCpp
#' @importFrom RcppParallel RcppParallelLibs
#' @importFrom stats dist dnorm pnorm rWishart rnorm runif
#' @importFrom utils read.table
#' @useDynLib mds
NULL
