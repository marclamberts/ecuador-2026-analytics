test_that("penalty_summary counts awarded/scored/missed", {
  m <- sample_match()
  events <- extract_set_pieces(m)
  summary <- penalty_summary(events)

  expect_equal(summary$awarded, 2)
  expect_equal(summary$scored, 1)
  expect_equal(summary$missed, 1)
  expect_equal(summary$saved, 0)
  expect_equal(summary$post, 0)
  expect_equal(summary$conversion_rate, 50.0)
})

test_that("penalty_summary handles no penalties", {
  summary <- penalty_summary(list())
  expect_equal(summary$awarded, 0)
  expect_true(is.na(summary$conversion_rate))
})
