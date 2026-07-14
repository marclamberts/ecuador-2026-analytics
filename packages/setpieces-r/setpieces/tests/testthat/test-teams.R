test_that("team_ids_from_filenames resolves teams seen across matches", {
  dir <- file.path(tempdir(), paste0("setpieces-test-", as.integer(Sys.time())))
  dir.create(dir, showWarnings = FALSE)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)

  match_a <- list(event = list(
    list(contestantId = "t_A", typeId = 1),
    list(contestantId = "t_B", typeId = 1)
  ))
  match_b <- list(event = list(
    list(contestantId = "t_A", typeId = 1),
    list(contestantId = "t_C", typeId = 1)
  ))
  match_c <- list(event = list(
    list(contestantId = "t_B", typeId = 1),
    list(contestantId = "t_C", typeId = 1)
  ))

  path_a <- file.path(dir, "2026-01-01_Team A - Team B.json")
  path_b <- file.path(dir, "2026-01-08_Team C - Team A.json")
  path_c <- file.path(dir, "2026-01-15_Team B - Team C.json")
  jsonlite::write_json(match_a, path_a, auto_unbox = TRUE)
  jsonlite::write_json(match_b, path_b, auto_unbox = TRUE)
  jsonlite::write_json(match_c, path_c, auto_unbox = TRUE)

  team_to_id <- team_ids_from_filenames(c(path_a, path_b, path_c))
  expect_equal(team_to_id[["Team A"]], "t_A")
  expect_equal(team_to_id[["Team B"]], "t_B")
  expect_equal(team_to_id[["Team C"]], "t_C")
})
