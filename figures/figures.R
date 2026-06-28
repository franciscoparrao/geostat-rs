#!/usr/bin/env Rscript
# Paper figures for geostat-rs (publishable methods only — no transport
# kriging). Reads figures/data/*.csv (from make_figure_data.py) and writes
# publication-quality PDFs to figures/.
#
# Run from the repo root: Rscript figures/figures.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

data_dir <- "figures/data"
options(stringsAsFactors = FALSE)

theme_paper <- theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 12),
    legend.position = "bottom",
    strip.background = element_rect(fill = "grey92", colour = NA)
  )

save_fig <- function(p, file, w = 6, h = 4) {
  ggsave(file.path("figures", file), p, width = w, height = h, device = "pdf")
  cat("wrote figures/", file, "\n", sep = "")
}

# --- Figure 1: gstat parity ------------------------------------------------
parity <- read.csv(file.path(data_dir, "parity.csv"))
lab <- by(parity, parity$method, function(d) {
  md <- max(abs(d$rust - d$gstat))
  data.frame(method = d$method[1], txt = sprintf("max abs diff = %.1e", md))
})
lab <- do.call(rbind, lab)
p1 <- ggplot(parity, aes(gstat, rust)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey60", linetype = 2) +
  geom_point(alpha = 0.25, size = 0.5, colour = "#2c6fbb") +
  geom_text(data = lab, aes(x = -Inf, y = Inf, label = txt),
            hjust = -0.1, vjust = 1.6, size = 3.2, inherit.aes = FALSE) +
  facet_wrap(~method, scales = "free") +
  labs(title = "geostat-rs vs gstat (Meuse grid)",
       x = "gstat prediction", y = "geostat-rs prediction") +
  theme_paper
save_fig(p1, "fig_parity.pdf", w = 7, h = 3.6)

# --- Figure 2: method comparison by VEcv (Meuse) ---------------------------
cmp <- read.csv(file.path(data_dir, "compare_vecv.csv"))
cmp$method <- factor(cmp$method, levels = cmp$method[order(cmp$vecv)])
p2 <- ggplot(cmp, aes(method, vecv, fill = method)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.1f", vecv)), hjust = -0.15, size = 3.3) +
  coord_flip(clip = "off") +
  scale_fill_brewer(palette = "Blues") +
  expand_limits(y = max(cmp$vecv) * 1.12) +
  labs(title = "Method comparison by leave-one-out VEcv (Meuse log-zinc)",
       x = NULL, y = "VEcv (%)") +
  theme_paper
save_fig(p2, "fig_compare.pdf", w = 6, h = 3.2)

# --- Figure 3: IDW power tuning --------------------------------------------
tune <- read.csv(file.path(data_dir, "idw_tune.csv"))
best <- tune[which.max(tune$vecv), ]
p3 <- ggplot(tune, aes(power, vecv)) +
  geom_line(colour = "#2c6fbb") +
  geom_point(colour = "#2c6fbb") +
  geom_point(data = best, colour = "#d1495b", size = 3) +
  geom_text(data = best, aes(label = sprintf("best: power %.1f", power)),
            vjust = -1, size = 3.2, colour = "#d1495b") +
  labs(title = "IDW power tuned by predictive accuracy (Meuse log-zinc)",
       x = "IDW power", y = "leave-one-out VEcv (%)") +
  theme_paper
save_fig(p3, "fig_idw_tune.pdf", w = 6, h = 3.6)

# --- Figure 4: Meuse anisotropy (variogram map + directional fit) ----------
vmap_path <- file.path(data_dir, "meuse_vmap.csv")
if (file.exists(vmap_path)) {
  vmap <- read.csv(vmap_path, na.strings = c("NA", ""))
  apar <- read.csv(file.path(data_dir, "meuse_aniso_params.csv"))
  adir <- read.csv(file.path(data_dir, "meuse_aniso_dir.csv"))
  afit <- read.csv(file.path(data_dir, "meuse_aniso_fit.csv"))
  adir$axis <- factor(adir$axis, levels = c("major", "minor"))
  afit$axis <- factor(afit$axis, levels = c("major", "minor"))
  axis_cols <- c(major = "#2c6fbb", minor = "#d1495b")

  # Panel a: 2-D variogram map; the dashed line marks the fitted major axis.
  az <- apar$azimuth[1]
  L <- max(abs(vmap$hx))
  rad <- az * pi / 180
  seg <- data.frame(
    x = -L * sin(rad), y = -L * cos(rad),
    xe = L * sin(rad), ye = L * cos(rad)
  )
  pa <- ggplot(vmap, aes(hx, hy, fill = gamma)) +
    geom_raster() +
    scico::scale_fill_scico(palette = "batlow", na.value = "grey90",
                            name = "semivariance") +
    geom_segment(data = seg, aes(x = x, y = y, xend = xe, yend = ye),
                 inherit.aes = FALSE, colour = "white", linetype = 2,
                 linewidth = 0.5) +
    annotate("label", x = 0.62 * L * sin(rad), y = 0.62 * L * cos(rad),
             label = sprintf("major axis\n%.0f°", az),
             size = 2.7, fill = "white", alpha = 0.7) +
    coord_fixed(expand = FALSE) +
    labs(x = expression(h[x] ~ "(m)"), y = expression(h[y] ~ "(m)")) +
    theme_paper +
    theme(legend.key.width = unit(1.2, "lines"))

  # Panel b: directional variograms with the fitted anisotropic model.
  rng <- sprintf("major range %.0f m,  ratio %.2f",
                 apar$major_range[1], apar$ratio[1])
  pb <- ggplot() +
    geom_line(data = afit, aes(h, gamma, colour = axis), linewidth = 0.6) +
    geom_point(data = adir, aes(h, gamma, colour = axis, size = n_pairs)) +
    scale_colour_manual(values = axis_cols, name = "axis",
                        labels = c("major (NE-SW)", "minor")) +
    scale_size_area(max_size = 3, guide = "none") +
    annotate("text", x = Inf, y = -Inf, label = rng,
             hjust = 1.05, vjust = -0.8, size = 2.8, colour = "grey30") +
    labs(x = "lag distance h (m)", y = "semivariance") +
    theme_paper

  p4a <- (pa | pb) +
    plot_annotation(tag_levels = "a", tag_suffix = ")") &
    theme(plot.tag = element_text(face = "bold", size = 11))
  save_fig(p4a, "fig_anisotropy.pdf", w = 7.2, h = 3.4)
} else {
  cat("anisotropy figure skipped (no data)\n")
}

# --- Figure 5: multi-element REE VEcv by method ----------------------------
me_path <- file.path(data_dir, "multielement.csv")
if (file.exists(me_path)) {
  me <- read.csv(me_path)
  me$element <- factor(me$element, levels = c("La", "Ce", "Nd", "Dy", "Y"))
  me$method <- factor(me$method, levels = c(
    "ordinary kriging", "regression kriging", "random forest", "RF + residual kriging"))
  p4 <- ggplot(me, aes(element, vecv, fill = method)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.75) +
    geom_hline(yintercept = 0, colour = "grey50", linewidth = 0.3) +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Rare-earth grade prediction: covariates + ML-geostat hybrid",
         subtitle = "Hold-out VEcv by element (Coquimbo tailings); host-mineral proxies",
         x = NULL, y = "VEcv (%)", fill = NULL) +
    theme_paper
  save_fig(p4, "fig_multielement.pdf", w = 7, h = 4)
} else {
  cat("multielement figure skipped (no data)\n")
}

cat("All figures rendered.\n")
