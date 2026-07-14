# Penalty award/conversion summaries.

#' Penalty award/conversion summary
#'
#' @param events A list of set-piece event records (from
#'   [extract_set_pieces()]). Only `kind == "penalty"` entries are
#'   counted.
#' @return A named list with `awarded`, `scored`, `saved`, `missed`,
#'   `post`, and `conversion_rate` (percentage, `NA` if none awarded).
#' @export
penalty_summary <- function(events) {
  penalties <- Filter(function(e) identical(e$kind, "penalty"), events)
  subtypes <- vapply(penalties, function(e) e$subtype, character(1))
  awarded <- length(subtypes)
  scored <- sum(subtypes == "goal")

  list(
    awarded = awarded,
    scored = scored,
    saved = sum(subtypes == "saved"),
    missed = sum(subtypes == "miss"),
    post = sum(subtypes == "post"),
    conversion_rate = if (awarded > 0) scored / awarded * 100.0 else NA_real_
  )
}
