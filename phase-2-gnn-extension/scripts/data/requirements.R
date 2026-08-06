required_packages <- c("reticulate", "simcausal")

missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    "Install missing R packages before data generation: ",
    paste(missing_packages, collapse = ", ")
  )
}
