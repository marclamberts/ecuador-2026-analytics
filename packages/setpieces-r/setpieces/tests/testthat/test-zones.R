test_that("classify_zone: short delivery", {
  expect_equal(classify_zone(end_x = 70.0, end_y = 50.0, start_y = 100.0), "short")
})

test_that("classify_zone: near/six, no mirroring needed", {
  expect_equal(classify_zone(end_x = 96.0, end_y = 30.0, start_y = 0.0), "near_six")
})

test_that("classify_zone: central/edge, mirrored", {
  # start_y >= 50 mirrors end_y -> my = 100 - 45 = 55 (central)
  expect_equal(classify_zone(end_x = 88.0, end_y = 45.0, start_y = 100.0), "central_edge")
})

test_that("zone_breakdown and zone_percentages", {
  m <- sample_match()
  corners <- extract_set_pieces(m, kinds = "corner")
  team_a <- Filter(function(e) e$team_id == "t_A", corners)

  counts <- zone_breakdown(team_a)
  expect_equal(as.integer(counts[["central_edge"]]), 1L)
  expect_equal(as.integer(counts[["near_six"]]), 1L)

  pct <- zone_percentages(team_a)
  expect_equal(as.numeric(pct[["central_edge"]]), 50.0)
  expect_equal(as.numeric(pct[["near_six"]]), 50.0)

  team_b <- Filter(function(e) e$team_id == "t_B", corners)
  counts_b <- zone_breakdown(team_b)
  expect_equal(as.integer(counts_b[["short"]]), 1L)
})
