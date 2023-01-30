
#include "AbstractMultiDimensionalScaling.hpp"
#include "NewMultiDimensionalScaling.hpp"

#include <Rcpp.h>
using namespace Rcpp;


// [[Rcpp::depends(RcppParallel,BH,RcppXsimd)]]
#include <RcppParallel.h>

// [[Rcpp::export]]
List rcpp_hello() {
  CharacterVector x = CharacterVector::create("foo", "bar");
  NumericVector y   = NumericVector::create(0.0, 1.0);
  List z            = List::create(x, y);
  return z;
}

using MdsSharedPtr = std::shared_ptr<mds::AbstractMultiDimensionalScaling>;

class MdsWrapper {
private:
  MdsSharedPtr mds;

public:
  MdsWrapper(MdsSharedPtr mds) : mds(mds) { }

  MdsSharedPtr& get() {
    return mds;
  }
};

using XPtrMdsWrapper = Rcpp::XPtr<MdsWrapper>;

MdsSharedPtr& parsePtr(SEXP sexp) {
  XPtrMdsWrapper ptr(sexp);
  if (!ptr) {
    Rcpp::stop("External pointer is uninitialized");
  }
  return ptr->get();
}

//' Create MDS engine object
//'
//' Helper function creates MDS engine object with given latent dimension, location count and various
//' implementation details. Called by \code{MassiveMDS::engineInitial()}.
//'
//' @param embeddingDimension Dimension of latent locations.
//' @param locationCount Number of locations and size of distance matrix.
//' @param tbb Number of CPU cores to be used.
//' @param simd For CPU implementation: no SIMD (\code{0}), SSE (\code{1}) or AVX (\code{2}).
//' @param truncation Likelihood includes truncation term? Defaults to \code{TRUE}.
//' @param gpu Which GPU to use? If only 1 available, use \code{gpu=1}. Defaults to \code{0}, no GPU.
//' @param single Set \code{single=1} if your GPU does not accommodate doubles.
//' @param bandwidth Number of pairwise couplings to include.
//' @return MDS engine object.
//'
//' @export
// [[Rcpp::export(createEngine)]]
Rcpp::List createEngine(int embeddingDimension, int locationCount, bool truncation, int tbb, int simd, int gpu, bool single, int bandwidth) {

  long flags = 0L;
  if (truncation) {
    flags |= mds::Flags::LEFT_TRUNCATION;
  }

  int deviceNumber = -1;
  int threads = 0;
  if (gpu > 0) {
    Rcout << "Running on GPU" << std::endl;
    flags |= mds::Flags::OPENCL;
    deviceNumber = gpu;
    if(single){
      flags |= mds::Flags::FLOAT;
      Rcout << "Single precision" << std::endl;
    }
  } else {
    Rcout << "Running on CPU" << std::endl;

#if RCPP_PARALLEL_USE_TBB
  if (tbb > 0) {
    threads = tbb;
    flags |= mds::Flags::TBB;
    std::shared_ptr<tbb::task_scheduler_init> task{nullptr};
    task = std::make_shared<tbb::task_scheduler_init>(threads);
  }
#endif

  if (simd == 1) {
    flags |= mds::Flags::SSE;
  } else if (simd == 2) {
    flags |= mds::Flags::AVX;
  }

  }

  auto mds = new MdsWrapper(mds::factory(embeddingDimension, locationCount,
                                         flags, deviceNumber, threads, bandwidth));
  XPtrMdsWrapper engine(mds);

  Rcpp::List list = Rcpp::List::create(
    Rcpp::Named("engine") = engine,
    Rcpp::Named("embeddingDimension") = embeddingDimension,
    Rcpp::Named("locationCount") = locationCount,
    Rcpp::Named("bandwidth") = bandwidth,
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
  return result;
}

// [[Rcpp::export(.getSumOfIncrements)]]
double getSumOfIncrements(SEXP sexp) {
  auto ptr = parsePtr(sexp);
  return ptr->getSumOfIncrements();
}
