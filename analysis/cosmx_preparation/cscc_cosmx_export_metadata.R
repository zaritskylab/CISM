#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: cscc_cosmx_export_metadata.R <Seurat_Object_cSCC.RDS> <metadata_csv_out>", call. = FALSE)
}

rds_path <- args[[1]]
metadata_out <- args[[2]]

object <- readRDS(rds_path)
metadata <- methods::slot(object, "meta.data")

metadata$cell_id <- rownames(metadata)

write.csv(metadata, metadata_out, row.names = FALSE, quote = TRUE)

cat("Exported metadata rows:", nrow(metadata), "\n")
cat("Exported metadata columns:", ncol(metadata), "\n")
cat("Output:", normalizePath(metadata_out, mustWork = FALSE), "\n")
