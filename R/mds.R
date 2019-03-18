
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

setPrecision <- function(engine, precision) {
  .setPrecision(engine$engine, precision)
  engine$precision <- precision
  return(engine)
}

setPairwiseData <- function(engine, data) {
  data <- as.vector(data)
  if (length(data) != engine$locationCount * engine$locationCount) {
    stop("Invalid data size")
  }
  .setPairwiseData(engine$engine, data)
  engine$dataInitialized <- TRUE
  return(engine)
}

updateLocations <- function(engine, locations) {
  locations <- as.vector(t(locations)) # C++ code assumes row-major
  if (length(locations) != engine$locationCount * engine$embeddingDimension) {
    stop("Invalid data size")
  }
  .updateLocations(engine$engine, locations)
  engine$locationsInitialized <- TRUE
  return(engine)
}
