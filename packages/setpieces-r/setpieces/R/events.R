# Extract structured set-piece events (corners, free kicks, throw-ins,
# penalties) out of a raw Opta match object.

#' Extract set-piece events from a match
#'
#' Each returned record has:
#' \itemize{
#'   \item `kind`: `"corner"`, `"free_kick"`, `"throw_in"`, or `"penalty"`
#'   \item `subtype`: `"delivery"` for corner/free_kick/throw_in deliveries;
#'     `"direct_shot_<goal|saved|post|miss>"` for direct free-kick shots;
#'     `"<goal|saved|post|miss>"` for penalties
#'   \item `team_id`, `player_name`, `period`, `minute`, `x`, `y`,
#'     `end_x`, `end_y`, `outcome`
#'   \item `event_index`: this event's position in [sorted_events()] for
#'     the same match -- needed to scan forward for second-ball contests
#' }
#'
#' @param match A parsed match object (from [load_match()]).
#' @param kinds Optional character vector restricting to a subset of
#'   `c("corner", "free_kick", "throw_in", "penalty")`.
#' @return A list of set-piece event records (each a named list).
#' @export
extract_set_pieces <- function(match, kinds = NULL) {
  events <- sorted_events(match)
  out <- list()

  for (i in seq_along(events)) {
    e <- events[[i]]
    type_id <- e$typeId
    if (is.null(type_id)) next
    qids <- qualifier_ids(e)

    kind <- NA_character_
    subtype <- NA_character_

    if (type_id == TYPE_PASS) {
      if (QUALIFIER_CORNER %in% qids) {
        kind <- "corner"; subtype <- "delivery"
      } else if (QUALIFIER_FREE_KICK %in% qids) {
        kind <- "free_kick"; subtype <- "delivery"
      } else if (QUALIFIER_THROW_IN %in% qids) {
        kind <- "throw_in"; subtype <- "delivery"
      }
    } else if (type_id %in% SHOT_TYPES) {
      shot_outcome <- unname(SHOT_OUTCOME_NAMES[as.character(type_id)])
      if (QUALIFIER_PENALTY %in% qids) {
        kind <- "penalty"; subtype <- shot_outcome
      } else if (QUALIFIER_FREE_KICK %in% qids) {
        kind <- "free_kick"; subtype <- paste0("direct_shot_", shot_outcome)
      }
    }

    if (is.na(kind)) next
    if (!is.null(kinds) && !(kind %in% kinds)) next

    if (identical(subtype, "delivery")) {
      end <- pass_end_xy(e)
    } else {
      end <- list(x = NA_real_, y = NA_real_)
    }

    record <- list(
      kind = kind, subtype = subtype,
      team_id = e$contestantId,
      player_name = if (is.null(e$playerName)) NA_character_ else e$playerName,
      period = as.integer(e$periodId), minute = minute_value(e),
      x = as.numeric(e$x), y = as.numeric(e$y),
      end_x = end$x, end_y = end$y,
      outcome = if (is.null(e$outcome)) NA_integer_ else as.integer(e$outcome),
      event_index = i
    )
    class(record) <- "setpiece_event"
    out[[length(out) + 1]] <- record
  }

  class(out) <- c("setpiece_events", "list")
  out
}

#' Convert a list of set-piece events to a data.frame
#'
#' @param events A list of set-piece event records (from
#'   [extract_set_pieces()]).
#' @return A data.frame with one row per event.
#' @export
set_pieces_to_df <- function(events) {
  if (length(events) == 0) {
    return(data.frame(
      kind = character(0), subtype = character(0), team_id = character(0),
      player_name = character(0), period = integer(0), minute = numeric(0),
      x = numeric(0), y = numeric(0), end_x = numeric(0), end_y = numeric(0),
      outcome = integer(0), event_index = integer(0),
      stringsAsFactors = FALSE
    ))
  }
  rows <- lapply(events, function(e) {
    data.frame(
      kind = e$kind, subtype = e$subtype, team_id = e$team_id,
      player_name = e$player_name, period = e$period, minute = e$minute,
      x = e$x, y = e$y, end_x = e$end_x, end_y = e$end_y,
      outcome = e$outcome, event_index = e$event_index,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}
