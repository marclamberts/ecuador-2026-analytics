# Coordinate helpers for Opta's 0-100 x 0-100 normalized pitch.

#' Real-world distance between two points on Opta's normalized pitch
#'
#' @param x0,y0,x1,y1 Coordinates in Opta's 0-100 normalized system.
#' @param pitch_length Pitch length in metres (default 105).
#' @param pitch_width Pitch width in metres (default 68).
#' @return Distance in metres.
#' @export
distance_m <- function(x0, y0, x1, y1, pitch_length = 105.0, pitch_width = 68.0) {
  dx <- (x1 - x0) / 100.0 * pitch_length
  dy <- (y1 - y0) / 100.0 * pitch_width
  sqrt(dx^2 + dy^2)
}

#' Match minute (with fractional seconds) for an Opta event
#'
#' @param event A single event (named list) from a match's `event` array.
#' @return Numeric minute, e.g. 45.5 for 45:30.
#' @export
minute_value <- function(event) {
  time_min <- if (is.null(event$timeMin)) 0 else event$timeMin
  time_sec <- if (is.null(event$timeSec)) 0 else event$timeSec
  as.numeric(time_min) + as.numeric(time_sec) / 60.0
}

#' qualifierId -> value for an Opta event
#'
#' @param event A single event (named list).
#' @return A named list keyed by qualifierId (as character).
#' @export
qualifier_map <- function(event) {
  q <- event$qualifier
  if (is.null(q) || length(q) == 0) return(list())
  ids <- vapply(q, function(x) as.character(x$qualifierId), character(1))
  vals <- lapply(q, function(x) x$value)
  stats::setNames(vals, ids)
}

#' qualifierId set for an Opta event
#'
#' @param event A single event (named list).
#' @return Integer vector of qualifierIds present on the event.
#' @export
qualifier_ids <- function(event) {
  q <- event$qualifier
  if (is.null(q) || length(q) == 0) return(integer(0))
  vapply(q, function(x) as.integer(x$qualifierId), integer(1))
}

#' End x/y of a pass event
#'
#' Falls back to the start location if the pass has no end-location
#' qualifiers (e.g. some fouled/blocked deliveries).
#'
#' @param event A single pass event (named list).
#' @return A list with `x` and `y` (both `NA_real_` if unavailable).
#' @export
pass_end_xy <- function(event) {
  qmap <- qualifier_map(event)
  ex <- qmap[[as.character(QUALIFIER_PASS_END_X)]]
  ey <- qmap[[as.character(QUALIFIER_PASS_END_Y)]]
  x <- if (is.null(ex)) event$x else as.numeric(ex)
  y <- if (is.null(ey)) event$y else as.numeric(ey)
  list(x = if (is.null(x)) NA_real_ else as.numeric(x),
       y = if (is.null(y)) NA_real_ else as.numeric(y))
}
