#!/usr/bin/env Rscript
# Schema-driven argument parser for omnibenchmark metrics.
# Reads a JSON schema and builds an argparse parser from it.
#
# Convention: all arguments are required; no defaults.

suppressPackageStartupMessages({
  library(argparse)
  library(jsonlite)
})

# Maps semantic JSON types to R argparse types.
.r_type <- function(t) {
  switch(t, path = "character", string = "character", t)
}

parse_args <- function() {
  schema_path <- file.path(getSrcDirectory(parse_args), "embedding.json")
  schema <- fromJSON(schema_path)
  parser <- ArgumentParser()

  for (i in seq_len(nrow(schema))) {
    row <- schema[i, ]
    dest <- if (!is.null(row$dest) && !is.na(row$dest)) row$dest else NULL
    parser$add_argument(
      row$flag,
      type = .r_type(row$type),
      dest = dest,
      help = row$help
    )
  }

  parser$parse_args()
}
