
#' Helper MDS log likelihood function
#'
#' Takes MDS engine object and returns log likelihood.
#'
#' @param engine An MDS engine object.
#' @return MDS log likelihood
#'
#' @export
getLogLikelihood <- function(engine) {

  if (!engine$dataInitialized) {
    stop("data are not set")
  }

  if (!engine$locationsInitialized) {
    stop("locations are not set")
  }

  if (is.null(engine$precision)) {
    stop("precision is not set")
  }

  sumOfIncrements <- .getSumOfIncrements(engine$engine)
  observationCount <- (engine$locationCount * (engine$locationCount - 1)) / 2;

  logLikelihood <- 0.5 * (log(engine$precision) - log(2 * pi)) * observationCount -
    sumOfIncrements

  return(logLikelihood)
}

#' Helper MDS log likelihood gradient function
#'
#' Takes MDS engine object and returns log likelihood gradient.
#'
#' @param engine An MDS engine object.
#' @return MDS log likelihood gradient.
#'
#' @export
getGradient <- function(engine) {

  if (!engine$dataInitialized) {
    stop("data are not set")
  }

  if (!engine$locationsInitialized) {
    stop("locations are not set")
  }

  if (is.null(engine$precision)) {
    stop("precision is not set")
  }

  matrix(.getLogLikelihoodGradient(engine$engine, engine$locationCount * engine$embeddingDimension),
         nrow = engine$locationCount, byrow = TRUE)
}

#' Deliver precision variable to MDS engine object
#'
#' Helper function delivers MDS likelihood precision to MDS engine object.
#'
#' @param engine MDS engine object.
#' @param precision MDS likelihood precision.
#' @return MDS engine object.
#'
#' @export
setPrecision <- function(engine, precision) {
  .setPrecision(engine$engine, precision)
  engine$precision <- precision
  return(engine)
}


#' Deliver distance matrix to MDS engine object
#'
#' Helper function delivers distance matrix to MDS engine object.
#'
#' @param engine MDS engine object.
#' @param data Distance matrix.
#' @return MDS engine object.
#'
#' @export
setPairwiseData <- function(engine, data) {
  data <- as.vector(data)
  if (length(data) != engine$locationCount * engine$locationCount) {
    stop("Invalid data size")
  }
  .setPairwiseData(engine$engine, data)
  engine$dataInitialized <- TRUE
  return(engine)
}

#' Deliver latent locations matrix to MDS engine object
#'
#' Helper function delivers latent locations matrix to MDS engine object.
#'
#' @param engine MDS engine object.
#' @param locations N by P matrix of N P-dimensional latent locations.
#' @return MDS engine object.
#'
#' @export
updateLocations <- function(engine, locations) {
  locations <- as.vector(t(locations)) # C++ code assumes row-major
  if (length(locations) != engine$locationCount * engine$embeddingDimension) {
    stop("Invalid data size")
  }
  .updateLocations(engine$engine, locations)
  engine$locationsInitialized <- TRUE
  return(engine)
}
