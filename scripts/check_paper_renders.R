# Execute the paper's R against the real data, without needing LaTeX.
#
# Two passes. First every chunk and inline expression is parsed, which catches a syntax error in
# seconds. Then the chunks are evaluated in order against data/, which catches the thing parsing
# cannot see: a column that is not there.
#
# This exists because check_paper_data.py missed exactly that. It compares the .qmd against the
# union of columns across all CSVs, so `n_items` looked present when the part-of-speech file had
# renamed it to std_n_items, and the whole llm object failed to build at render time. Executing
# the setup found it immediately.
#
#   Rscript scripts/check_paper_renders.R
#
# Exits non-zero on a parse error or a failed data chunk. Figure and table chunks are reported but
# not fatal: those can fail on a missing plotting package, which says nothing about the data.

qmd <- "paper/separability.qmd"
if (!file.exists(qmd)) { cat("run me from the repo root\n"); quit(status = 1) }
txt <- readLines(qmd, warn = FALSE)
starts <- grep("^```[{]r", txt)
ends <- grep("^```$", txt)

# ---- pass 1: parse -------------------------------------------------------------------------
bad <- 0L; n <- 0L
for (s in starts) {
  e <- ends[ends > s][1]; if (is.na(e)) next
  n <- n + 1L
  msg <- tryCatch({ parse(text = paste(txt[(s + 1):(e - 1)], collapse = "\n")); NULL },
                  error = function(err) conditionMessage(err))
  if (!is.null(msg)) { bad <- bad + 1L; cat(sprintf("  PARSE chunk line %d: %s\n", s, msg)) }
}
inline <- unlist(regmatches(txt, gregexpr("`r [^`]+`", txt)))
for (x in inline) {
  msg <- tryCatch({ parse(text = substr(x, 4, nchar(x) - 1)); NULL },
                  error = function(err) conditionMessage(err))
  if (!is.null(msg)) { bad <- bad + 1L; cat(sprintf("  PARSE inline %s: %s\n", x, msg)) }
}
cat(sprintf("%d chunks, %d inline expressions parsed; %d failed\n", n, length(inline), bad))
if (bad > 0) quit(status = 1)

# ---- pass 2: evaluate against the data -----------------------------------------------------
owd <- setwd(dirname(qmd)); on.exit(setwd(owd))
env <- new.env(); fatal <- 0L; soft <- 0L
for (s in starts) {
  e <- ends[ends > s][1]; if (is.na(e)) next
  code <- paste(txt[(s + 1):(e - 1)], collapse = "\n")
  msg <- tryCatch({ eval(parse(text = code), envir = env); NULL },
                  error = function(err) conditionMessage(err))
  if (is.null(msg)) next
  # a chunk that only draws or tabulates is reported, not fatal
  drawing <- grepl("ggplot|kbl\\(|kable", code)
  if (drawing) { soft <- soft + 1L; cat(sprintf("  (figure/table) line %d: %s\n", s, substr(msg, 1, 110)))
  } else { fatal <- fatal + 1L; cat(sprintf("  DATA CHUNK line %d: %s\n", s, substr(msg, 1, 200))) }
}
cat(sprintf("evaluated against data/: %d data-chunk failures, %d figure/table notes\n",
            fatal, soft))

# ---- pass 3: evaluate the inline expressions -------------------------------------------------
# Parsing an inline expression proves it is syntactically valid, not that the function it calls
# exists. A prose number calling a helper that was renamed parses fine and fails at render, which
# is how `ldrange` for `ldrng` survived two passes. These are the numbers the prose asserts, so a
# missing one is a wrong sentence rather than a missing figure.
bad_inline <- 0L
for (x in inline) {
  code <- substr(x, 4, nchar(x) - 1)
  v <- tryCatch(eval(parse(text = code), envir = env),
                error = function(e) structure(conditionMessage(e), class = "sepErr"))
  msg <- NULL
  if (inherits(v, "sepErr")) msg <- substr(v, 1, 80)
  else if (length(v) != 1L) msg <- sprintf("not scalar (length %d)", length(v))
  else if (is.atomic(v) && is.na(v)) msg <- "NA"
  else if (is.numeric(v) && !is.finite(v)) msg <- "not finite"
  if (!is.null(msg)) {
    bad_inline <- bad_inline + 1L
    cat(sprintf("  INLINE %-52s -> %s
", substr(code, 1, 52), msg))
  }
}
cat(sprintf("inline expressions evaluated: %d, %d problematic
", length(inline), bad_inline))
fatal <- fatal + bad_inline

for (nm in c("llm", "llm_deep", "toy", "toy_cap")) {
  if (exists(nm, envir = env)) {
    d <- get(nm, envir = env)
    cat(sprintf("  %-9s %6d rows x %3d cols\n", nm, nrow(d), ncol(d)))
  } else cat(sprintf("  %-9s NOT BUILT\n", nm))
}
quit(status = if (fatal > 0) 1 else 0)
