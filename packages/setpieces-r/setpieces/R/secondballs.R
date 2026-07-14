# Second-ball contest detection after a team's own set pieces.
#
# For each qualifying delivery, look at the next events (within a short
# time window) for contested actions (aerial duel, tackle, interception,
# clearance, ball recovery). The *second* such contested action is
# treated as "the second ball" -- the first is usually just the target
# player's initial header/contact, the second is who actually comes
# away with the loose ball. This is very often well outside the box,
# since a strong clearance can travel 40+ metres before the second ball
# is won.

FINAL_THIRD_START <- 200.0 / 3.0

#' Second-ball contests following a team's own set pieces
#'
#' @param events `sorted_events(match)` for the same match `set_pieces`
#'   was built from -- contest detection scans forward from each
#'   delivery's `event_index` in this list.
#' @param set_pieces A list of set-piece event records (from
#'   [extract_set_pieces()]) for the same match.
#' @param team_id The delivering team's contestantId.
#' @param window_events Max number of subsequent events to scan.
#' @param window_minutes Max time (in minutes) after the delivery to
#'   keep scanning (default 12 seconds).
#' @param max_clean_passes_between Max number of clean (successful)
#'   passes allowed between the first and second contested action
#'   before treating the chain as broken.
#' @param final_third_start X-coordinate a delivery must reach to
#'   qualify (default: start of the final third).
#' @return A list of contest records, each with `delivery` (the
#'   originating set-piece event), `x`, `y`, `won` (logical), and
#'   `winner_player`.
#' @export
find_second_ball_contests <- function(events, set_pieces, team_id,
                                       window_events = 5, window_minutes = 12 / 60,
                                       max_clean_passes_between = 1,
                                       final_third_start = FINAL_THIRD_START) {
  results <- list()

  for (sp in set_pieces) {
    if (!identical(sp$team_id, team_id)) next
    if (is.na(sp$end_x) || sp$end_x < final_third_start) next

    i <- sp$event_index
    t0 <- minute_value(events[[i]])

    window <- list()
    j <- i + 1
    while (j <= length(events) && length(window) < window_events) {
      if (minute_value(events[[j]]) - t0 > window_minutes) break
      window[[length(window) + 1]] <- events[[j]]
      j <- j + 1
    }
    if (length(window) == 0) next

    first_idx <- NA_integer_
    for (k in seq_along(window)) {
      if (!is.null(window[[k]]$typeId) && window[[k]]$typeId %in% CONTESTED_TYPES) {
        first_idx <- k
        break
      }
    }
    if (is.na(first_idx)) next

    second_ball <- NULL
    clean_passes <- 0
    if (first_idx < length(window)) {
      for (k in (first_idx + 1):length(window)) {
        ev <- window[[k]]
        if (!is.null(ev$typeId) && ev$typeId %in% CONTESTED_TYPES) {
          second_ball <- ev
          break
        }
        if (!is.null(ev$typeId) && ev$typeId == TYPE_PASS && !is.null(ev$outcome) && ev$outcome == 1) {
          clean_passes <- clean_passes + 1
          if (clean_passes > max_clean_passes_between) break
        }
      }
    }
    if (is.null(second_ball) || is.null(second_ball$x) || is.null(second_ball$y)) next

    won <- identical(second_ball$contestantId, team_id)
    results[[length(results) + 1]] <- list(
      delivery = sp, x = as.numeric(second_ball$x), y = as.numeric(second_ball$y),
      won = won,
      winner_player = if (is.null(second_ball$playerName)) NA_character_ else second_ball$playerName
    )
  }

  results
}
