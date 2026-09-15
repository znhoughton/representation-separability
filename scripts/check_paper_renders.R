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

# ---- pass 0: escapes a shell heredoc can eat ------------------------------------------------
# Writing LaTeX through a heredoc turns a backslash and a letter into the control character it
# names, so "\ref{" arrives as a line break followed by "ef{" and "\alpha" as a bell. The damage
# is silent and survives a render. A line break before "ef{" is never legitimate in this paper,
# and neither is a control character.
#
# The characters are built from their code points rather than written literally, because a
# literal carriage return in this file does not survive being read and rewritten by a tool that
# normalises line endings, and a guard that corrupts itself is worse than no guard.
raw <- paste(readLines(qmd, warn = FALSE), collapse = "\n")
eaten <- sum(gregexpr("\nef[{]", raw, fixed = TRUE)[[1]] > 0)
ctrl <- 0L
for (cp in c(7L, 8L, 11L, 12L, 13L)) {
  hits <- gregexpr(intToUtf8(cp), raw, fixed = TRUE)[[1]]
  ctrl <- ctrl + sum(hits > 0)
}
if (eaten + ctrl > 0) {
  cat(sprintf("  MANGLED ESCAPES: %d line-break-before-ef{ and %d control characters\n",
              eaten, ctrl))
  cat("  a backslash was eaten writing LaTeX through a heredoc; repair before rendering\n")
  quit(status = 1)
}

# ---- pass 0b: floats nobody points at -------------------------------------------------------
# A figure or table the prose never cites is either a float the reader is never sent to, or a
# cross-reference lost in an edit. Both have happened here: rewriting a section dropped the only
# pointer to a table, and to an entire appendix.
#
# The brace and backslash are built from code points. Written literally they are escapes, and an
# escape in this file is one shell heredoc away from being eaten.
all_txt <- paste(readLines(qmd, warn = FALSE), collapse = " ")
OB <- intToUtf8(123)
BS <- intToUtf8(92)
pick <- function(prefix) {
  m <- gregexpr(paste0("(?<=", prefix, ")(fig|tbl)-[A-Za-z0-9-]+"), all_txt, perl = TRUE)
  unlist(regmatches(all_txt, m))
}
lab <- unique(c(pick("label: "), pick(paste0(OB, "#"))))
cit <- unique(c(pick("@"), pick(paste0("ref", OB))))
orphan <- setdiff(lab, cit)
if (length(orphan)) {
  cat(sprintf("  UNCITED: %d float(s) the prose never points at: %s\n",
              length(orphan), paste(orphan, collapse = ", ")))
}

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


# ---- pass 4: document order -------------------------------------------------------------------
# Passes 2 and 3 evaluate every chunk first and every inline expression afterwards, so an inline
# expression sitting ABOVE the chunk that defines what it reads still passes. Quarto evaluates top
# to bottom and fails such a render with "object not found", which is how a table chunk moved below
# the prose citing it reached the author. This walks the file in order instead, evaluating each
# chunk and each inline expression as it is met. The working directory is already paper/ from
# pass 2, so it is not changed again here.
ord_env <- new.env()
bad_order <- 0L
c_end <- vapply(starts, function(s) {
  e <- ends[ends > s][1]
  if (is.na(e)) s else e
}, numeric(1))
in_chunk <- rep(FALSE, length(txt))
for (k in seq_along(starts)) in_chunk[starts[k]:c_end[k]] <- TRUE

k <- 1L
for (ln in seq_along(txt)) {
  if (k <= length(starts) && ln == starts[k]) {
    e <- c_end[k]
    if (e > ln + 1) {
      try(eval(parse(text = paste(txt[(ln + 1):(e - 1)], collapse = "\n")), envir = ord_env),
          silent = TRUE)
    }
    k <- k + 1L
    next
  }
  if (in_chunk[ln]) next
  for (x in unlist(regmatches(txt[ln], gregexpr("`r [^`]+`", txt[ln])))) {
    code <- substr(x, 4, nchar(x) - 1)
    msg <- tryCatch({ eval(parse(text = code), envir = ord_env); NULL },
                    error = function(e) conditionMessage(e))
    if (!is.null(msg) && grepl("not found", msg)) {
      bad_order <- bad_order + 1L
      cat(sprintf("  OUT OF ORDER line %d: %s -> %s\n", ln, substr(code, 1, 44),
                  substr(msg, 1, 56)))
    }
  }
}
cat(sprintf("document order: %d expression(s) used above the chunk that defines them\n",
            bad_order))
fatal <- fatal + bad_order

for (nm in c("llm", "llm_deep", "toy", "toy_cap")) {
  if (exists(nm, envir = env)) {
    d <- get(nm, envir = env)
    cat(sprintf("  %-9s %6d rows x %3d cols\n", nm, nrow(d), ncol(d)))
  } else cat(sprintf("  %-9s NOT BUILT\n", nm))
}
quit(status = if (fatal > 0) 1 else 0)
