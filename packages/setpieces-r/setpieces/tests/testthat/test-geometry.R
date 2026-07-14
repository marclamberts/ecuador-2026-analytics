test_that("distance_m: zero distance", {
  expect_equal(distance_m(50, 50, 50, 50), 0.0)
})

test_that("distance_m: full pitch length", {
  expect_equal(distance_m(0, 50, 100, 50, pitch_length = 105.0, pitch_width = 68.0), 105.0)
})

test_that("distance_m: full pitch width", {
  expect_equal(distance_m(50, 0, 50, 100, pitch_length = 105.0, pitch_width = 68.0), 68.0)
})
