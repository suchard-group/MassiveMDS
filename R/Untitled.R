
computeLoglikelihood <- function(data, locations, precision, truncation = TRUE, gradient = FALSE) {
  # serially computed MDS log likelihood and log likelihood gradient
  # data is distance matrix
  # locations are positions in latent space

  locationCount <- dim(data)[1]
  sd <- 1 / sqrt(precision)
  logLikelihood <- 0
  gradlogLikelihood <- 0*locations
  for (i in 1:locationCount) {
    for (j in i:locationCount) {
      if (i != j) {
        mean <- as.numeric(dist(rbind(locations[i,], locations[j,])))
        if (gradient) {
          if(truncation) {
            gradlogLikelihood[i,] <- gradlogLikelihood[i,] - ((mean-data[i,j])*precision+dnorm(mean/sd)/(sd*pnorm(mean/sd)))*
              (locations[i,]-locations[j,])/mean
            gradlogLikelihood[j,] <- gradlogLikelihood[j,] - ((mean-data[i,j])*precision+dnorm(mean/sd)/(sd*pnorm(mean/sd)))*
              (locations[j,]-locations[i,])/mean
          } else {
            gradlogLikelihood[i,] <- gradlogLikelihood[i,] + (data[i,j]-mean)*precision*(locations[i,]-locations[j,])/mean
            gradlogLikelihood[j,] <- gradlogLikelihood[j,] + (data[i,j]-mean)*precision*(locations[j,]-locations[i,])/mean
          }
        } else {
          logLikelihood <- logLikelihood + dnorm(data[i, j], mean = mean, sd = sd, log = TRUE)
          if (truncation) {
            logLikelihood <- logLikelihood - log(pnorm(mean/sd))
          }
        }
      }
    }
  }

  if (gradient) {
    return(gradlogLikelihood)
  }
  else {
    return(logLikelihood)
  }
}


dmatrixnorm <- function(X, Mu = NULL, U, V, Uinv, Vinv, gradient=FALSE) {
  # log density and its gradient for matrix normal prior on latent locations (X)
  # n is number of objects
  # p is latent dimension
  # U is nxn covariance
  # V is pxp covariance
  n <- dim(X)[1]
  p <- dim(X)[2]

  if (!is.null(Mu)) {
    X <- X - Mu
  }

  if (gradient) {
    result <- -0.5*(t(Vinv %*% t(X) %*% Uinv) + Uinv %*% X %*% Vinv)
    return(unname(result))
  }
  else {
  product <- Vinv %*% t(X) %*% Uinv %*% X
  exponent <- -0.5 * sum(diag(product))

  logDetV <- determinant(V, logarithm = TRUE)
  logDetU <- determinant(U, logarithm = TRUE)

  result <- exponent - (n * p / 2) * log(2 * pi) -
    (n / 2) * logDetV$modulus - (p / 2) * logDetU$modulus

  return(as.vector(result)) # remove attributes
  }
}

test <- function(  locationCount =10,threads=0,simd=0) {
  # function compares serially and parallel-ly computed log likelihoods and gradients,
  # returns log likelihoods (should be equal) and distance between gradients (should be 0)
  # threads is number of CPU cores used
  # simd = 0, 1, 2 for no simd, SSE, and AVX, respectively
  embeddingDimension <- 2
  truncation <- FALSE

  data <- matrix(rnorm(n = locationCount * locationCount, sd = 2),
                 ncol = locationCount, nrow = locationCount)
  data <- data * data    # Make positive
  data <- data + t(data) # Make symmetric
  diag(data) <- 0        # Make pairwise distance

  locations <- matrix(rnorm(n = embeddingDimension * locationCount, sd = 1),
                      ncol = embeddingDimension, nrow = locationCount)

  cat("no trunc\n")
  engine <- mds::createEngine(embeddingDimension, locationCount, truncation, threads, simd)
  engine <- mds::setPairwiseData(engine, data)
  engine <- mds::updateLocations(engine, locations)

  cat("logliks\n")
  engine <- mds::setPrecision(engine, 2.0)
  print(mds::getLogLikelihood(engine))
  print(computeLoglikelihood(data, locations, 2.0, truncation))

  engine <- mds::setPrecision(engine, 0.5)
  print(mds::getLogLikelihood(engine))
  print(computeLoglikelihood(data, locations, 0.5, truncation))

  cat("grads (max error)\n")
  engine <- mds::setPrecision(engine, 2.0)
  print(max(abs(mds::getGradient(engine) -
                  computeLoglikelihood(data, locations, 2.0, truncation,gradient = TRUE))))

  engine <- mds::setPrecision(engine, 0.5)
  print(max(abs(mds::getGradient(engine) -
                  computeLoglikelihood(data, locations, 0.5, truncation,gradient = TRUE))))

  truncation <- TRUE

  engine <- mds::createEngine(embeddingDimension, locationCount, truncation, threads, simd)
  engine <- mds::setPairwiseData(engine, data)
  engine <- mds::updateLocations(engine, locations)

  cat("logliks\n")
  engine <- mds::setPrecision(engine, 2.0)
  print(mds::getLogLikelihood(engine))
  print(computeLoglikelihood(data, locations, 2.0, truncation))

  engine <- mds::setPrecision(engine, 0.5)
  print(mds::getLogLikelihood(engine))
  print(computeLoglikelihood(data, locations, 0.5, truncation))

  cat("grads (max error)\n")
  engine <- mds::setPrecision(engine, 2.0)
  print(max(abs(mds::getGradient(engine) -
                  computeLoglikelihood(data, locations, 2.0, truncation,gradient = TRUE))))

  engine <- mds::setPrecision(engine, 0.5)
  print(max(abs(mds::getGradient(engine) -
                  computeLoglikelihood(data, locations, 0.5, truncation,gradient = TRUE))))
}


timeTest <- function(locationCount=5000, maxIts=1, threads=0, simd=0) {
  # function returns length of time to compute log likelihood and gradient
  # threads is number of CPU cores used
  # simd = 0, 1, 2 for no simd, SSE, and AVX, respectively
  embeddingDimension <- 2
  truncation <- TRUE

  data <- matrix(rnorm(n = locationCount * locationCount, sd = 2),
                 ncol = locationCount, nrow = locationCount)
  data <- data * data    # Make positive
  data <- data + t(data) # Make symmetric
  diag(data) <- 0        # Make pairwise distance

  locations <- matrix(rnorm(n = embeddingDimension * locationCount, sd = 1),
                      ncol = embeddingDimension, nrow = locationCount)
  engine <- mds::createEngine(embeddingDimension, locationCount, truncation, threads, simd)
  engine <- mds::setPairwiseData(engine, data)
  engine <- mds::updateLocations(engine, locations)
  engine <- mds::setPrecision(engine, 2.0)

  ptm <- proc.time()
  for(i in 1:maxIts){
    mds::getLogLikelihood(engine)
    mds::getGradient(engine)
  }
  proc.time() - ptm
}



install_dependencies <- function() {
  # We require the development version of OutbreakTools from github
  devtools::install_github("thibautjombart/OutbreakTools")
  # TODO: remove dependence on no longer maintained package
}

# example <- function() {
#   # Read BEAST tree
#   tree <- OutbreakTools::read.annotated.nexus("inst/extdata/large.trees")
#   N <- length(tree$tip.label)
#
#   # Get tree VCV conditional on root
#   treeLength <- sum(tree$edge.length)
#   treeVcv <- caper::VCV.array(tree) / treeLength
#   class(treeVcv) <- "matrix"
#
#   # Integrate out fully conjugate root
#   priorSampleSize <- 10
#   treeVcv <- treeVcv + matrix(1 / priorSampleSize, ncol = N, nrow = N)
#
#   # Get trait VCV
#   tmp <- unlist(tree$tree.annotations$precision)
#   traitVcv <- solve(matrix(c(tmp[1], tmp[2], tmp[2], tmp[3]), ncol = 2))
#
#   # Get initial tip locations
#   locations <- t(sapply(1:length(tree$tip.label),
#                         function(i) {
#                           index <- which(tree$edge[,2] == i)
#                           unlist(tree$annotations[[index]]$loc)
#                         }))
#   rownames(locations) <- tree$tip.label
#   P <- dim(locations)[2]
#
#   # Compute log prior density (matrix-normal distribution)
#   logPrior <- dmatrixnorm(X = locations,
#                           U = treeVcv,
#                           V = traitVcv)
#
#   # Read data for MDS likelihood
#   data <- read.table("inst/extdata/h3_Deff.txt", sep = "\t", header = TRUE, row.names = 1)
#   stopifnot(dim(data)[1] == N)
#   permutation <- sapply(1:N,
#                         function(i) {
#                           which(tree$tip.label[i] == rownames(data))
#                         })
#   data <- data[permutation, permutation]
#
#   # Specifiy precision for MDS
#   precision <- 1.470221879
#
#   # Compute the log likelihood and gradient (slowly)
#   system.time(
#     logLikelihood1 <- computeLoglikelihood(data = data, locations = locations,
#                                           precision = precision,
#                                           truncation = FALSE) # Truncation is not tested yet
#   )
#
#   system.time(
#     gradient1 <- computeLoglikelihood(data, locations, precision, gradient = TRUE)
#   )
#
#   # Build reusable object
#   truncation <- FALSE
#   engine <- mds::createEngine(embeddingDimension = P,
#                                     locationCount = N, truncation = truncation)
#   # Set data once
#   engine <- mds::setPairwiseData(engine, as.matrix(data))
#
#   # Call every time locations change
#   engine <- mds::updateLocations(engine, locations)
#
#   # Call every time precision changes
#   engine <- mds::setPrecision(engine, precision = precision)
#
#   # Compute the log likelihood (faster, cached)
#   system.time(
#     logLikelihood2 <- mds::getLogLikelihood(engine)
#   )
#
#   # Compute gradient (faster, but not yet cached)
#   system.time(
#     gradient2 <- mds::getGradient(engine)
#   )
# }


readbeast <- function(file = "large", priorRootSampleSize = 0.001) {
  # Read BEAST tree
  tree <- OutbreakTools::read.annotated.nexus(
    paste0("inst/extdata/", file, ".trees"))
  N <- length(tree$tip.label)

  # Get tree VCV conditional on root
  treeLength <- sum(tree$edge.length)
  treeVcv <- caper::VCV.array(tree) / treeLength
  class(treeVcv) <- "matrix"

  # Integrate out fully conjugate root
  treeVcv <- treeVcv + matrix(1 / priorRootSampleSize, ncol = N, nrow = N)

  # Get trait VCV
  tmp <- unlist(tree$tree.annotations$precision)
  traitVcv <- solve(matrix(c(tmp[1], tmp[2], tmp[2], tmp[3]), ncol = 2))

  # Get initial tip locations
  locations <- t(sapply(1:length(tree$tip.label),
                        function(i) {
                          index <- which(tree$edge[,2] == i)
                          unlist(tree$annotations[[index]]$loc)
                        }))
  rownames(locations) <- tree$tip.label

  log <- read.table(
    paste0("inst/extdata/", file, ".log"),
    sep = "\t", header = TRUE, skip = 3)
  #P <- dim(locations)[2]
  return(list(locations = locations,
              treeVcv = treeVcv,
              traitVcv = traitVcv,
              d0 = 2,
              traitT0 = diag(c(1,1)),
              log = log))
}


getdata <- function(N, locations) { # TEMP EDIT: ADD locations param
  # Read data for MDS likelihood
  data <- read.table("inst/extdata/h3_Deff.txt", sep = "\t", header = TRUE, row.names = 1)
  stopifnot(dim(data)[1] == N)
  permutation <- sapply(1:N,
                        function(i) {
                          #which(tree$tip.label[i] == rownames(data))
                          which(rownames(locations)[i] == rownames(data)) # TEMP EDIT
                        })
  inverse_permutation <- sapply(1:N,
                                function(i) {
                                  which(permutation == i)
                                })
  data <- data[permutation, permutation]
  attr(data, "inverse_permutation") <- inverse_permutation
  return(data)
}


engineInitial <- function(data,locations,N,P,
                          precision = 1.470222,threads,simd,truncation) {

  # Build reusable object
  engine <- mds::createEngine(embeddingDimension = P,
                                    locationCount = N, truncation = truncation, tbb = threads, simd=simd)
  # Set data once
  engine <- mds::setPairwiseData(engine, as.matrix(data))

  # Call every time locations change
  engine <- mds::updateLocations(engine, locations)

  # Call every time precision changes
  engine <- mds::setPrecision(engine, precision = precision)

  return(engine)
}

Potential <- function(engine,locations,treeVcv,traitVcv,treePrec,traitPrec,gradient=FALSE) {
    # HMC potential (log posterior) and gradient
    # treeVcv is n locations x n locations
    # traitVcv is n latent dimensions x n latent dimensions
    if (gradient) {
    logPriorGrad <- dmatrixnorm(X = locations, U = treeVcv, V = traitVcv,
                                Uinv=treePrec,Vinv=traitPrec,gradient=gradient)
    logLikelihoodGrad <- mds::getGradient(engine)

    return(-(logPriorGrad + logLikelihoodGrad))
  }
  else {
    logPrior <- dmatrixnorm(X = locations, U = treeVcv, V = traitVcv,
                            Uinv=treePrec, Vinv=traitPrec)
    logLikelihood <- mds::getLogLikelihood(engine)

    return(-(logPrior+logLikelihood))
  }
}


hmcsampler <- function(n_iter,
                       BurnIn,
                       file = "large",
                       learnPrec=FALSE,               # learn MDS likelihood precision?
                       learnTraitPrec=FALSE,          # learn pxp latent precision?
                       Trajectory = 0.2,              # length of HMC proposal trajectory
                       traitInvWeight = 1,
                       randomizeInitialState = FALSE,
                       priorRootSampleSize = 0.001,
                       mdsPrecision = 1.470222,       # if learnPrec=FALSE then mdsPrecision=1.47
                       threads=1,                     # number of CPU cores
                       simd=0,                        # simd = 0, 1, 2 for no simd, SSE, and AVX, respectively
                       treeCov=FALSE,                 # if treeCov=TRUE, tree-based nxn precision used
                       truncation=TRUE) {

  # Set up the parameters
  NumOfIterations = n_iter
  # HMC tuning parameters
  NumOfLeapfrog = 20
  StepSize = Trajectory/NumOfLeapfrog

  # Allocate output space
  LocationSaved = list()
  Target = vector()
  savedLikEvals <- rep(0,n_iter)

  # Read BEAST tree
  beast <- readbeast(file = file, priorRootSampleSize)
  locations <- beast$locations
  N <- dim(locations)[1]
  P <- dim(locations)[2]

  if (randomizeInitialState) {
    set.seed(666)
    saveNames <- rownames(locations)
    locations <- matrix(rnorm(N * P, 0, 1), nrow = N, ncol = P)
    rownames(locations) <- saveNames
  }

  # Read data for MDS likelihood
  data <- getdata(N, locations)   # TEMP EDIT: ADD locations param

  # Build reusable object to compute Loglikelihood (gradient)
  engine <- engineInitial(data,locations,N,P, mdsPrecision,threads,simd,truncation)

  Accepted = 0;
  Proposed = 0;
  likEvals = 0;

  # if we want to learn scalar precision
  if (learnPrec) {
    precision <- rep(0, n_iter)
    precision[1] <- engine$precision

    acceptPrec <- 0
    proposePrec <- 0
  } else {
    precision <- rep(engine$precision, n_iter)
  }

  # Incorporate tree structure into nxn covariance?
  if (treeCov) { # if yes
    treePrec <- solve(beast$treeVcv)   # Need inverse Vcovs
  } else {       # if not
    beast$treeVcv <- diag(N)           # set Vcov and precision to identity
    treePrec      <- diag(N)
  }

  # if we want to learn P-dim precision (traitPrec)
  if(learnTraitPrec){
    traitPrec <- array(0,dim=c(n_iter,P,P))
    traitPrec[1,,] <- solve(beast$traitVcv) #rWishart(1, beast$d0 + N,solve(beast$traitT0 + t(locations)%*%treePrec%*%locations))
  }else{
    # for tidiness, fill out array with fixed matrix (so same code can
    # handle both learning and not learning traitPrec)
    trait_prec <- solve(beast$traitVcv)
    traitPrec <- array(0,dim=c(n_iter,P,P))
    for(j in 1: n_iter) traitPrec[j,,] <- trait_prec
  }

  # Initialize the location
  CurrentLocation = locations;

  # if we want to learn traitPrec use iterate else use fixed matrix
  # to evaluate potential
  engine <- mds::updateLocations(engine, CurrentLocation)
  engine <- mds::setPrecision(engine, precision[1])
  CurrentU = Potential(engine,locations,treeVcv=beast$treeVcv,
                      traitVcv = solve(traitPrec[1,,]),
                      treePrec = treePrec, traitPrec = traitPrec[1,,])

  cat(paste0('Initial log-likelihood: ', mds::getLogLikelihood(engine), '\n'))

  # track number of likelihood evaluations
  likEvals = likEvals + 1;

  # Perform Hamiltonian Monte Carlo
  for (Iteration in 1:NumOfIterations) {

    ProposedLocation = CurrentLocation
    engine <- mds::updateLocations(engine, ProposedLocation)
    engine <- mds::setPrecision(engine, precision[Iteration])

    # Sample the marginal momentum
    CurrentMomentum = MASS::mvrnorm(N,rep(0,P),matrix(c(1,0,0,1),2,2))
    ProposedMomentum = CurrentMomentum

    Proposed = Proposed + 1

    # Simulate the Hamiltonian Dynamics
    for (StepNum in 1:NumOfLeapfrog) {
      ProposedMomentum = ProposedMomentum - StepSize/2 * Potential(engine,ProposedLocation,beast$treeVcv,solve(traitPrec[Iteration,,]),
                                                                   treePrec=treePrec, traitPrec=traitPrec[Iteration,,], gradient=T)
      likEvals = likEvals + 1;

      ProposedLocation = ProposedLocation + StepSize * ProposedMomentum
      engine <- mds::updateLocations(engine, ProposedLocation)
      ProposedMomentum = ProposedMomentum - StepSize/2 * Potential(engine,ProposedLocation,beast$treeVcv,solve(traitPrec[Iteration,,]),
                                                                   treePrec=treePrec, traitPrec=traitPrec[Iteration,,], gradient=T)
      likEvals = likEvals + 1;
    }

    ProposedMomentum = - ProposedMomentum

    # Compute the Potential
    ProposedU = Potential(engine,ProposedLocation,beast$treeVcv,solve(traitPrec[Iteration,,]),
                          treePrec=treePrec, traitPrec=traitPrec[Iteration,,])
    likEvals = likEvals + 1;

    # Compute the Hamiltonian
    CurrentH = CurrentU + sum(CurrentMomentum^2)/2
    ProposedH = ProposedU + sum(ProposedMomentum^2)/2

    # Accept according to ratio
    Ratio = - ProposedH + CurrentH
    if (is.finite(Ratio) & (Ratio > min(0,log(runif(1))))) {
      CurrentLocation = ProposedLocation
      CurrentU = ProposedU
      Accepted = Accepted + 1
    }

    # Save if sample is required
    if (Iteration > BurnIn) {
      LocationSaved[[Iteration-BurnIn]] = CurrentLocation
      Target[Iteration-BurnIn] = CurrentU
      savedLikEvals[Iteration-BurnIn] = likEvals
    }

    # Show acceptance rate every 20 iterations
    if (Iteration %% 20 == 0) {
      cat(Iteration, "iterations completed. HMC acceptance rate: ",Accepted/Proposed,"\n")

      Proposed = 0
      Accepted = 0
    }

    # Start timer after burn-in
    if (Iteration == BurnIn) { # If BurnIn > 0
      cat("Burn-in complete, now drawing samples ...\n")
      timer = proc.time()
    }
    if (BurnIn==0 & Iteration==1) { # If BurnIn = 0
      cat("Burn-in complete, now drawing samples ...\n")
      timer = proc.time()
    }

    # MH step for residual precision
    if (learnPrec) {
      prec_star <- abs(runif(1, precision[Iteration] - .01, precision[Iteration] + .01)) # draw from uniform
      engine <- mds::setPrecision(engine, prec_star)
      ProposedU = Potential(engine,CurrentLocation,beast$treeVcv,solve(traitPrec[Iteration,,]),
                            treePrec = treePrec, traitPrec = traitPrec[Iteration,,])

      Ratio = -ProposedU + CurrentU
      proposePrec <- proposePrec + 1

      if (is.finite(Ratio) & (Ratio > min(0,log(runif(1))))) {
        precision[Iteration + 1] <- prec_star
        acceptPrec <- acceptPrec + 1
      } else {
        precision[Iteration + 1] <- precision[Iteration]
        engine <- mds::setPrecision(engine, precision[Iteration])
      }

      if (Iteration %% 20 == 0) { # print MH acceptances
        cat(Iteration, "iterations completed. Prec acceptance rate: ",acceptPrec/proposePrec,"\n")

        proposePrec = 0
        acceptPrec = 0
      }
    }

    # Gibbs draw for traitPrec if learnTraitPrec == TRUE
    if (Iteration < n_iter & learnTraitPrec) {
      if (Iteration %% traitInvWeight == 0) {
        traitPrec[Iteration + 1,,] <- rWishart(1, beast$d0 + N, solve(beast$traitT0 + t(CurrentLocation) %*% treePrec %*% CurrentLocation))
      } else {
        traitPrec[Iteration + 1,,] <- traitPrec[Iteration,,]
      }
    }
  }

  time = proc.time() - timer
  acprat = dim(LocationSaved[!duplicated(LocationSaved)])[1]/(NumOfIterations-BurnIn)

  # if learn everything, return ...
  if(learnPrec & learnTraitPrec){
    return(list(samples = LocationSaved, target = Target, Time = time, acprat = acprat,
                likEvals = savedLikEvals, precision=precision,
                traitPrec=traitPrec))
  }

  # if learn prec, return ...
  if(learnPrec){
    return(list(samples = LocationSaved, target = Target, Time = time, acprat = acprat,
                likEvals = savedLikEvals, precision=precision))
  }

  # if learn trait prec, return ...
  if(learnTraitPrec){
    return(list(samples = LocationSaved, target = Target, Time = time, acprat = acprat,
                likEvals = savedLikEvals, traitPrec=traitPrec))
  }

  # only HMC, return ...
  return(list(samples = LocationSaved, target = Target, Time = time, acprat = acprat,
              likEvals = savedLikEvals))

}


eSliceSampler <- function(n_iter, BurnIn, mdsPrecision = 1.470222, randomizeInitialState = FALSE){
  # Set up the parameters
  NumOfIterations = n_iter
  # BurnIn = floor(0.2*NumOfIterations)

  # Allocate output space
  LocationSaved = list()
  Target = vector()
  savedLikEvals <- rep(0,n_iter)

  # Read BEAST tree
  beast <- readbeast()
  locations <- beast$locations
  N <- dim(locations)[1]
  P <- dim(locations)[2]

  if (randomizeInitialState) {
    set.seed(666)
    saveNames <- rownames(locations)
    locations <- matrix(rnorm(N * P, 0, 1), nrow = N, ncol = P)
    rownames(locations) <- saveNames
  }

  # Read data for MDS likelihood
  data <- getdata(N, locations)   # TEMP EDIT: ADD locations param
  cat('Data generated\n')

  # Build reusable object to compute Loglikelihood (gradient)
  engine <- engineInitial(data,locations,N,P, mdsPrecision)
  cat('Initial engine built\n')

  # Initialize the location
  likEvals <- 0;
  CurrentLocation <- locations # QUESTION: is this an initialization at truth?
  CurrentLogLik <- mds::getLogLikelihood(engine)
  likEvals = likEvals + 1;

  cat(paste0('Initial log-likelihood: ', CurrentLogLik, '\n'))

  # Need inverse Vcovs for efficient computation of potential
  treePrec  <- solve(beast$treeVcv)
  traitPrec <- solve(beast$traitVcv)

  # Get Vcv matrix Cholesky factors (nec. for drawing from prior)
  Chol_treeVcv  <- chol(beast$treeVcv)
  Chol_traitVcv <- chol(beast$traitVcv)

  # Perform elliptical slice sampling
  for (Iteration in 1:NumOfIterations) {

    # Random draw from prior
    nu <- matrix(rnorm(N*P),N,P)
    nu <- t(Chol_treeVcv) %*% nu %*% Chol_traitVcv

    # Log likelihood threshold
    u <- runif(1)
    threshold <- CurrentLogLik + log(u)

    # Draw bracket
    theta <- runif(n=1,min=0,max=2*pi)
    thetaMax <- theta
    thetaMin <- thetaMax - 2*pi

    # While proposed state not accepted
    flag <- TRUE
    while (flag == TRUE) {
      # Get proposal and evaluate logLik
      ProposedLocation <- CurrentLocation*cos(thetaMax) +
        nu*sin(thetaMax)
      engine <- mds::updateLocations(engine, ProposedLocation)
      ProposedLogLik <- mds::getLogLikelihood(engine)
      likEvals = likEvals + 1;

      if (ProposedLogLik > threshold) {
        # Accept ProposedLocation
        CurrentLocation <- ProposedLocation # Update
        CurrentLogLik <- ProposedLogLik
        CurrentU <- Potential(engine,CurrentLocation,beast$treeVcv,beast$traitVcv,
                              treePrec = treePrec, traitPrec = traitPrec) # Get log probability for comparison to HMC

        flag <- FALSE
      } else {
        # Shrink the bracket and try a new point
        if (theta < 0){
          thetaMin <- theta
        } else {
          thetaMax <- theta
        }
        theta <- runif(1, thetaMin, thetaMax)
      }
    }

    # Save if sample is required
    if (Iteration > BurnIn) {
      LocationSaved[[Iteration-BurnIn]] = CurrentLocation
      Target[Iteration-BurnIn] = CurrentU
      savedLikEvals[Iteration-BurnIn] = likEvals
    }

    # Show acceptance rate every 100 iterations
    if (Iteration%%20 == 0) {
      cat(Iteration, " iterations completed\n")
    }

    # Start timer after burn-in
    if (Iteration == BurnIn) { # If BurnIn > 0
      cat("Burn-in complete, now drawing samples ...\n")
      timer = proc.time()
    }
    if (BurnIn==0 & Iteration==1) { # If BurnIn = 0
      cat("Burn-in complete, now drawing samples ...\n")
      timer = proc.time()
    }


  }

  time = proc.time() - timer
  acprat = dim(LocationSaved[!duplicated(LocationSaved)])[1]/(NumOfIterations-BurnIn)
  return(list(samples = LocationSaved, target = Target, Time = time, acprat = acprat,
              likEvals = savedLikEvals))

}

# check_initial_state <- function(file = "large",
#                                 priorRootSampleSize = 0.001,
#                                 mdsPrecision = 1.470222) {
#   beast <- readbeast(file = file, priorRootSampleSize)
# browser()
#   x <- dmatrixnorm(X = beast$locations, U = beast$treeVcv, V = beast$traitVcv,
#                    Uinv = solve(beast$treeVcv), Vinv = solve(beast$traitVcv), gradient = FALSE)
#
#   x_gradient <- dmatrixnorm(X = beast$locations, U = beast$treeVcv, V = beast$traitVcv,
#                             Uinv = solve(beast$treeVcv), Vinv = solve(beast$traitVcv), gradient = TRUE)
#
#   locations <- beast$locations
#   N <- dim(locations)[1]
#   P <- dim(locations)[2]
#
#   # Read data for MDS likelihood
#   data <- getdata(N, locations)
#   inverse_permutation <- attr(data, "inverse_permutation")
#
#   # Build reusable object to compute Loglikelihood (gradient)
#   engine <- engineInitial(data,locations,N,P, mdsPrecision)
#
#   y <- mds::getLogLikelihood(engine)
#   y_gradient <- mds::getGradient(engine)
#
#   return(list(estimates = c(x,beast$log$loc.traitLikelihood, y, beast$log$mdsLikelihood),
#               x_gradient = x_gradient[inverse_permutation,],
#               y_gradient = y_gradient[inverse_permutation,]))
# }

# printRandomPrecisionForBEAST <- function() {
#   set.seed(666)
#
#   Sigma <- rWishart(1, 10, diag(nrow = 10))
#   P <- solve(Sigma[,,1])
#
#   apply(P, MARGIN = 1, function(x) {
#     line <- paste0("\t\t<parameter value=\"",
#            paste(as.list(x), collapse = " "), "\"/>\n")
#     cat(line, sep = "")
#     line
#     })
# }
#
# .hide <- function() {
#   chk <- check_initial_state("h3_large_geo_BMDS_Country_HMC")
#   chk$estimates
#   chk$y_gradient[1:4,]
#   chk$x_gradient[1:4,]
# }
