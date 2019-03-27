
#include "AbstractMultiDimensionalScaling.hpp"
#include "NewMultiDimensionalScaling.hpp"

#include <Rcpp.h>
using namespace Rcpp;


// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>

// [[Rcpp::export]]
List rcpp_hello() {
  CharacterVector x = CharacterVector::create("foo", "bar");
  NumericVector y   = NumericVector::create(0.0, 1.0);
  List z            = List::create(x, y);
  return z;
}


typedef Rcpp::XPtr<mds::AbstractMultiDimensionalScaling> EnginePtr;

EnginePtr parsePtr(SEXP sexp) {
  EnginePtr ptr(sexp);
  if (!ptr) {
    Rcpp::stop("External pointer is uninitialized");
  }
  return ptr;
}

// [[Rcpp::export(createEngine)]]
Rcpp::List createEngine(int embeddingDimension, int locationCount, bool truncation, int tbb, int simd) {

  long flags = 0L;
  if (truncation) {
    flags |= mds::Flags::LEFT_TRUNCATION;
  }

  int deviceNumber = -1;

  int threads = 0;

  auto newMdsObj = new mds::NewMultiDimensionalScaling<mds::DoubleNoSimdTypeInfo,mds::CpuAccumulate>(embeddingDimension, locationCount, flags, threads);

#if RCPP_PARALLEL_USE_TBB
  std::shared_ptr<tbb::task_scheduler_init> task{nullptr};

  if (tbb > 0) {
    threads = tbb;
    if (threads != 0) {
      flags |= mds::Flags::TBB;
      task = std::make_shared<tbb::task_scheduler_init>(threads);
#define TBB_SWITCH
    }
  }
#else
  tbb = 0;
#endif

 if ( !(simd == 0 || (simd == 1 || simd == 2)) ) {
   stop("simd = 0 (no simd) 1 (sse) or 2 (avx)");
 }
 if (simd == 0) {
   std::cout << "Using no SIMD" << std::endl;

#ifdef TBB_SWITCH
   auto newMdsObj = new mds::NewMultiDimensionalScaling<mds::DoubleNoSimdTypeInfo,mds::TbbAccumulate>(embeddingDimension, locationCount, flags, threads);
#endif

 }
 if (simd == 1) {
#ifdef USE_SSE
   flags |= mds::Flags::SSE;
   std::cout << "Using SSE vectorization" << std::endl;

#ifdef TBB_SWITCH
   auto newMdsObj = new mds::NewMultiDimensionalScaling<mds::DoubleSseTypeInfo,mds::TbbAccumulate>(embeddingDimension, locationCount, flags, threads);
#else
   auto newMdsObj = new mds::NewMultiDimensionalScaling<mds::DoubleSseTypeInfo,mds::CpuAccumulate>(embeddingDimension, locationCount, flags, threads);
#endif

#else
   std::cout << "Using no SIMD" << std::endl;
#endif
 }
 if (simd == 2) {
#ifdef USE_AVX
   flags |= mds::Flags::AVX;
   std::cout << "Using AVX vectorization" << std::endl;

#ifdef TBB_SWITCH
   auto newMdsObj = new mds::NewMultiDimensionalScaling<mds::DoubleAvxTypeInfo,mds::TbbAccumulate>(embeddingDimension, locationCount, flags, threads);
#else
   auto newMdsObj = new mds::NewMultiDimensionalScaling<mds::DoubleAvxTypeInfo,mds::CpuAccumulate>(embeddingDimension, locationCount, flags, threads);
#endif

#else
   std::cout << "Using no SIMD" << std::endl;
#endif
 }

 EnginePtr engine( newMdsObj );

  Rcpp::List list = Rcpp::List::create(
    Rcpp::Named("engine") = engine,
    Rcpp::Named("embeddingDimension") = embeddingDimension,
    Rcpp::Named("locationCount") = locationCount,
    Rcpp::Named("dataInitialzied") = false,
    Rcpp::Named("locationsInitialized") = false,
    Rcpp::Named("truncation") = truncation,
    Rcpp::Named("threads") = threads,
    Rcpp::Named("deviceNumber") = deviceNumber,
    Rcpp::Named("flags") = flags
  );

  return list;
}

// [[Rcpp::export(.setPairwiseData)]]
void setPairwiseData(SEXP sexp,
                    std::vector<double>& data) {
  auto ptr = parsePtr(sexp);
  ptr->setPairwiseData(&data[0], data.size());
}

// [[Rcpp::export(.updateLocations)]]
void updateLocations(SEXP sexp,
                     std::vector<double>& locations) {
  auto ptr = parsePtr(sexp);
  ptr->updateLocations(-1, &locations[0], locations.size());
}

// [[Rcpp::export(.setPrecision)]]
void setPrecision(SEXP sexp, double precision) {
  auto ptr = parsePtr(sexp);
  ptr->setParameters(&precision, 1L);
}

// [[Rcpp::export(.getLogLikelihoodGradient)]]
std::vector<double> getLogLikelihoodGradient(SEXP sexp, size_t len) {
  auto ptr = parsePtr(sexp);
  std::vector<double> result(len);
  ptr->getLogLikelihoodGradient(&result[0], len);
  return std::move(result);
}

// [[Rcpp::export(.getSumOfIncrements)]]
double getSumOfIncrements(SEXP sexp) {
  auto ptr = parsePtr(sexp);
  return ptr->getSumOfIncrements();
}
