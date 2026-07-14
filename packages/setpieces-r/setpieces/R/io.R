# Loading Opta MA1-style match JSON files.

#' Load a single match JSON file
#'
#' @param path Path to a match JSON file (must contain an `event` array).
#' @return A named list parsed from JSON.
#' @export
load_match <- function(path) {
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

#' Load several match JSON files, in the given order
#'
#' @param paths Character vector of file paths.
#' @return A list of parsed match objects.
#' @export
load_matches <- function(paths) {
  lapply(paths, load_match)
}

#' This match's events, in the order they actually happened
#'
#' Regular time and extra time only (periodId 1 or 2), ordered by
#' period, minute, then eventId.
#'
#' @param match A parsed match object (from [load_match()]).
#' @return A list of event objects, chronologically ordered.
#' @export
sorted_events <- function(match) {
  events <- Filter(function(e) !is.null(e$periodId) && e$periodId %in% c(1, 2), match$event)
  if (length(events) == 0) return(list())

  periods <- vapply(events, function(e) as.integer(e$periodId), integer(1))
  minutes <- vapply(events, minute_value, numeric(1))
  event_ids <- vapply(events, function(e) if (is.null(e$eventId)) 0L else as.integer(e$eventId), integer(1))

  ord <- order(periods, minutes, event_ids)
  events[ord]
}
