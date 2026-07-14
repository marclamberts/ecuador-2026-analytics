test_that("second ball won by delivering team", {
  m <- sample_match()
  events <- sorted_events(m)
  set_pieces <- extract_set_pieces(m)
  contests <- find_second_ball_contests(events, set_pieces, "t_A")

  expect_length(contests, 1)
  contest <- contests[[1]]
  expect_true(contest$won)
  expect_equal(contest$winner_player, "A. Second Ball Winner")
  expect_equal(contest$x, 60.0)
  expect_equal(contest$y, 40.0)
  expect_equal(contest$delivery$kind, "corner")
})

test_that("no contests when nothing follows the delivery", {
  m <- sample_match()
  events <- sorted_events(m)
  set_pieces <- extract_set_pieces(m)
  # t_B's only delivery is the last event in the match, so there's
  # nothing after it to scan for a contest
  contests <- find_second_ball_contests(events, set_pieces, "t_B")
  expect_length(contests, 0)
})
