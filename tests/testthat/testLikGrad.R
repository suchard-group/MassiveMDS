library(MassiveMDS)

context("testLikGrad.R")

test_that("parallel likelihood is same as serial", {
  skip_on_cran()
  expect_equal(test(threads=1), test(threads = 2))
})
