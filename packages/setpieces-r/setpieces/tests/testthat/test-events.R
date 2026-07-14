test_that("extracts all set-piece kinds", {
  m <- sample_match()
  events <- extract_set_pieces(m)
  kinds <- sort(vapply(events, function(e) e$kind, character(1)))
  expect_equal(kinds, sort(c("corner", "corner", "corner", "free_kick", "throw_in", "penalty", "penalty")))
})

test_that("filters by kind", {
  m <- sample_match()
  corners <- extract_set_pieces(m, kinds = "corner")
  expect_length(corners, 3)
  expect_true(all(vapply(corners, function(e) e$kind == "corner", logical(1))))
})

test_that("corner fields are populated correctly", {
  m <- sample_match()
  corners <- extract_set_pieces(m, kinds = "corner")
  first <- corners[[1]]
  expect_equal(first$team_id, "t_A")
  expect_equal(first$player_name, "A. Corner Taker")
  expect_equal(first$subtype, "delivery")
  expect_equal(first$end_x, 88.0)
  expect_equal(first$end_y, 45.0)
})

test_that("penalty subtypes are goal/miss", {
  m <- sample_match()
  penalties <- extract_set_pieces(m, kinds = "penalty")
  subtypes <- sort(vapply(penalties, function(e) e$subtype, character(1)))
  expect_equal(subtypes, c("goal", "miss"))
})

test_that("event_index matches sorted_events", {
  m <- sample_match()
  events <- sorted_events(m)
  set_pieces <- extract_set_pieces(m)
  for (sp in set_pieces) {
    raw <- events[[sp$event_index]]
    expect_equal(raw$contestantId, sp$team_id)
  }
})

test_that("set_pieces_to_df builds a data.frame", {
  m <- sample_match()
  events <- extract_set_pieces(m)
  df <- set_pieces_to_df(events)
  expect_equal(nrow(df), length(events))
  expect_true(all(c("kind", "team_id", "end_x", "end_y") %in% names(df)))
})
