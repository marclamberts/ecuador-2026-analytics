# Delivery-zone classification for corners and free kicks.
#
# Deliveries are mirrored onto a single attacking side (near-post low,
# far-post high) so left- and right-side set pieces combine into one
# near/central/far-post x six-yard/edge-of-box 6-zone breakdown, plus a
# "short" bucket for deliveries that don't reach the box at all.

#' Default zone geometry parameters
#'
#' A named list with `box_front_x`, `six_yard_front_x`, `near_cut`, and
#' `far_cut`, all in Opta's 0-100 normalized pitch units.
#'
#' @export
default_zone_params <- list(
  box_front_x = 83.0,
  six_yard_front_x = 94.0,
  near_cut = 41.0,
  far_cut = 59.0
)

#' Classify a single delivery's end location into a zone key
#'
#' @param end_x,end_y Delivery end location (0-100 normalized).
#' @param start_y Y-coordinate the delivery was taken from (used to
#'   mirror right-side deliveries onto the left).
#' @param params Zone geometry parameters, see [default_zone_params].
#' @return `"short"` if the ball never reached the penalty box,
#'   otherwise a `"<near|central|far>_<six|edge>"` string.
#' @export
classify_zone <- function(end_x, end_y, start_y, params = default_zone_params) {
  if (is.na(end_x) || is.na(end_y)) return(NA_character_)
  if (end_x < params$box_front_x) return("short")

  my <- if (start_y >= 50) 100 - end_y else end_y
  col <- if (my < params$near_cut) "near" else if (my > params$far_cut) "far" else "central"
  row <- if (end_x >= params$six_yard_front_x) "six" else "edge"
  paste(col, row, sep = "_")
}

#' Raw counts per zone key across a set of deliveries
#'
#' @param events A list of set-piece event records (from
#'   [extract_set_pieces()]). Events without an end location are
#'   skipped.
#' @param params Zone geometry parameters, see [default_zone_params].
#' @return A named integer vector, zone key -> count.
#' @export
zone_breakdown <- function(events, params = default_zone_params) {
  zones <- vapply(events, function(e) {
    if (is.na(e$end_x) || is.na(e$end_y)) return(NA_character_)
    classify_zone(e$end_x, e$end_y, e$y, params)
  }, character(1))
  zones <- zones[!is.na(zones)]
  if (length(zones) == 0) return(stats::setNames(integer(0), character(0)))
  table(zones)
}

#' Each zone's share of all classified deliveries, as a percentage
#'
#' @inheritParams zone_breakdown
#' @return A named numeric vector, zone key -> percentage.
#' @export
zone_percentages <- function(events, params = default_zone_params) {
  counts <- zone_breakdown(events, params)
  total <- sum(counts)
  if (total == 0) return(stats::setNames(numeric(0), character(0)))
  counts / total * 100.0
}
