#!/usr/bin/env bash
# REPORT_PCA (global gather): render pca_report.qmd over the PCA_RESULTS table
# (pca_results_tsv, every PCA_M table concatenated) into one self-contained HTML file.
#   ./render.sh --output_dir D --name N [--output report_html=PATH] --from-files lineage.json
# --from-files (ob api 0.8.0, lineage.json v2) lists every member's files by output id.
set -euo pipefail

out_dir="" name="" out="" manifest=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) out_dir="$2"; shift 2 ;;
    --name) name="$2"; shift 2 ;;
    --output) [[ "$2" == report_html=* ]] && out="${2#report_html=}"; shift 2 ;;
    --from-files) manifest="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$manifest" ]] || { echo "no --from-files given" >&2; exit 2; }
mapfile -t files < <(Rscript -e 'm <- jsonlite::fromJSON(commandArgs(TRUE)[1], simplifyVector = FALSE)$members
writeLines(unlist(lapply(m, function(x) x$files$pca_results_tsv)))' "$manifest")
[[ ${#files[@]} -eq 1 ]] || { echo "expected one pca_results_tsv in $manifest, got ${#files[@]}" >&2; exit 2; }
out="${out:-$out_dir/pca.html}"

# quarto renders next to the .qmd: work in a scratch copy, keep the module dir clean
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$(dirname "$0")/pca_report.qmd" "$work/"
printf '%s\n' "${files[@]}" > "$work/inputs.txt"
(cd "$work" && quarto render pca_report.qmd --to html)
mkdir -p "$(dirname "$out")"
mv "$work/pca_report.html" "$out"
echo "wrote $out from ${files[0]}"
