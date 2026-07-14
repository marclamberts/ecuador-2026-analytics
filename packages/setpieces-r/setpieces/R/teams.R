# Convenience helper to map team names to Opta contestantIds from this
# repo's match-file naming convention: "YYYY-MM-DD_Home Team - Away Team.json".
#
# This is optional -- every other function in the package works
# directly off contestantId strings, so use this only if your file
# layout follows that convention.

FILENAME_RE <- "^\\d{4}-\\d{2}-\\d{2}_(.+) - (.+)\\.json$"

#' Infer team name -> contestantId from match filenames
#'
#' A team name resolves only if the intersection of contestantIds seen
#' across every match file where that team appears (as home or away)
#' narrows to exactly one ID.
#'
#' @param paths Character vector of match JSON file paths, named
#'   `YYYY-MM-DD_Home Team - Away Team.json`.
#' @return A named list, team display name -> contestantId.
#' @export
team_ids_from_filenames <- function(paths) {
  team_cid_sets <- list()

  for (path in paths) {
    base <- basename(path)
    m <- regmatches(base, regexec(FILENAME_RE, base))[[1]]
    if (length(m) < 3) next
    home <- m[2]
    away <- m[3]

    match <- load_match(path)
    cids <- unique(vapply(match$event, function(e) {
      if (is.null(e$contestantId)) NA_character_ else e$contestantId
    }, character(1)))
    cids <- cids[!is.na(cids)]

    for (team in c(home, away)) {
      if (is.null(team_cid_sets[[team]])) team_cid_sets[[team]] <- list()
      team_cid_sets[[team]][[length(team_cid_sets[[team]]) + 1]] <- cids
    }
  }

  result <- list()
  for (team in names(team_cid_sets)) {
    sets <- team_cid_sets[[team]]
    inter <- Reduce(intersect, sets)
    if (length(inter) == 1) result[[team]] <- inter[1]
  }
  result
}
