
#include "AbstractMultiDimensionalScaling.hpp"
#include "NewMultiDimensionalScaling.hpp"

#include <Rcpp.h>
using namespace Rcpp;

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
Rcpp::List createEngine(int embeddingDimension, int locationCount, bool truncation, int threads) {

  long flags = 0L;
  if (truncation) {
    flags |= mds::LEFT_TRUNCATION;
  }

  EnginePtr engine(
      new mds::NewMultiDimensionalScaling<mds::DoubleNoSimdTypeInfo,mds::CpuAccumulate>(embeddingDimension, locationCount, flags, threads)
  );

  Rcpp::List list = Rcpp::List::create(
    Rcpp::Named("engine") = engine,
    Rcpp::Named("embeddingDimension") = embeddingDimension,
    Rcpp::Named("locationCount") = locationCount,
    Rcpp::Named("dataInitialzied") = false,
    Rcpp::Named("locationsInitialized") = false,
    Rcpp::Named("truncation") = truncation,
    Rcpp::Named("threads") = threads
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
