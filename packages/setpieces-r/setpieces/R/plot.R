# Optional plotting helpers. Requires ggplot2:
#   install.packages("ggplot2")

.check_ggplot2 <- function() {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Plotting requires the ggplot2 package. Install it with: install.packages(\"ggplot2\")", call. = FALSE)
  }
}

.zone_rects <- function(params) {
  six_front <- params$six_yard_front_x
  data.frame(
    col = c("near", "central", "far", "near", "central", "far"),
    row = c("edge", "edge", "edge", "six", "six", "six"),
    y0 = c(0, params$near_cut, params$far_cut, 0, params$near_cut, params$far_cut),
    y1 = c(params$near_cut, params$far_cut, 100, params$near_cut, params$far_cut, 100),
    x0 = c(params$box_front_x, params$box_front_x, params$box_front_x, six_front, six_front, six_front),
    x1 = c(six_front, six_front, six_front, 100, 100, 100),
    stringsAsFactors = FALSE
  )
}

#' Attacking-third pitch with the 6-zone delivery grid, coloured by percentage
#'
#' @param zone_pct A named numeric vector, zone key
#'   (`"<near|central|far>_<six|edge>"`) -> percentage -- the output of
#'   [zone_percentages()]. The `"short"` key, if present, is ignored (it
#'   has no location to draw).
#' @param params Zone geometry parameters, see [default_zone_params].
#' @param title Optional plot title.
#' @return A ggplot object.
#' @export
plot_zone_grid <- function(zone_pct, params = default_zone_params, title = NULL) {
  .check_ggplot2()

  rects <- .zone_rects(params)
  rects$zone <- paste(rects$col, rects$row, sep = "_")
  rects$pct <- vapply(rects$zone, function(z) {
    if (is.na(zone_pct[z])) 0 else as.numeric(zone_pct[z])
  }, numeric(1))
  rects$label <- sprintf("%.0f%%", rects$pct)

  ggplot2::ggplot(rects) +
    ggplot2::geom_rect(ggplot2::aes(xmin = .data$y0, xmax = .data$y1, ymin = .data$x0, ymax = .data$x1, fill = .data$pct),
                        color = "#11161f", linewidth = 1) +
    ggplot2::geom_text(ggplot2::aes(x = (.data$y0 + .data$y1) / 2, y = (.data$x0 + .data$x1) / 2, label = .data$label),
                        color = "white", fontface = "bold") +
    ggplot2::scale_fill_viridis_c(name = "%") +
    ggplot2::coord_fixed(xlim = c(0, 100), ylim = c(50, 100)) +
    ggplot2::labs(title = title, x = NULL, y = NULL) +
    ggplot2::theme_void() +
    ggplot2::theme(legend.position = "right")
}

#' Full-pitch scatter of second-ball contests
#'
#' @param contests A list of contest records (from
#'   [find_second_ball_contests()]).
#' @param title Optional plot title.
#' @return A ggplot object.
#' @export
plot_second_ball_map <- function(contests, title = NULL) {
  .check_ggplot2()

  df <- do.call(rbind, lapply(contests, function(c) {
    data.frame(x = c$x, y = c$y, won = c$won)
  }))
  if (is.null(df)) df <- data.frame(x = numeric(0), y = numeric(0), won = logical(0))

  ggplot2::ggplot(df, ggplot2::aes(x = .data$y, y = .data$x, color = .data$won)) +
    ggplot2::geom_point(size = 3, alpha = 0.8) +
    ggplot2::scale_color_manual(values = c(`TRUE` = "#ffc247", `FALSE` = "#9aa4b2"),
                                 labels = c(`TRUE` = "Won", `FALSE` = "Lost"), name = NULL) +
    ggplot2::coord_fixed(xlim = c(0, 100), ylim = c(0, 100)) +
    ggplot2::labs(title = title, x = NULL, y = NULL) +
    ggplot2::theme_void()
}
