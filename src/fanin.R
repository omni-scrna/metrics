# Reading an ob gather record (`--from-files`, lineage.json v2, ob api 0.8) in R.
#
# Module-owned, not under src/common/ (which `sync` overwrites): the R
# counterpart of from_files() / member_parameters() in src/common/cli.py, for
# any R gather in this module. Every member lists its files by output id and
# its full lineage, so nothing here parses ob's output paths.

# Members that carry output `id` (e.g. "embedding_tsv"), in manifest order.
manifest_members <- function(path, id) {
  members <- jsonlite::fromJSON(path, simplifyVector = FALSE)$members
  Filter(function(m) !is.null(m$files[[id]]), members)
}

# A parameter value as one table cell: scalars as they are, lists and mappings
# as JSON text, JSON null as NA.
.cell <- function(v) {
  if (is.null(v)) return(NA)
  if (is.atomic(v) && length(v) == 1L) return(v)
  as.character(jsonlite::toJSON(v, auto_unbox = TRUE))
}

# One member's lineage as named table columns, in the shape PCA_M writes
# (pca_metrics.py: dataset, norm, feat, pca, pca_params, <stage>_<parameter>):
#   dataset                 the DATA module id
#   <stage>                 "<module>.<params hash>" per upstream stage ("<module>" without parameters)
#   <as>_stage, <as>, <as>_params
#                           the member's own node (its last lineage entry): stage,
#                           module, params hash. `as` names what is gathered, e.g.
#                           "embedding", since members may come from several stages.
#   <stage>_<parameter>     every parameter value, as cli.member_parameters names them
member_lineage_cols <- function(member, as) {
  lin <- member$lineage
  own <- lin[[length(lin)]]
  data <- Find(function(e) e$stage == "DATA", lin)
  cols <- list(dataset = data$module)
  for (e in lin[-length(lin)]) {
    if (e$stage == "DATA") next
    cols[[tolower(e$stage)]] <- if (is.null(e$params)) e$module else paste0(e$module, ".", e$params)
  }
  cols[[paste0(as, "_stage")]] <- own$stage
  cols[[as]] <- own$module
  cols[[paste0(as, "_params")]] <- .cell(own$params)
  for (e in lin) {
    for (k in names(e$parameters)) {
      cols[[paste0(tolower(e$stage), "_", k)]] <- .cell(e$parameters[[k]])
    }
  }
  cols
}
