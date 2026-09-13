"""Does the paper read anything the data does not carry, and does the data carry anything the
paper never looks at?

The third joint. `check_emits` guarantees the measure's fields reach a writer, and the round-trip
test guarantees a writer's fields reach a CSV. Neither says anything about whether the .qmd can
find what it asks for, or whether a number we paid to compute is going unreported.

Both directions matter, and for opposite reasons:

  paper -> data   a column the .qmd names but no CSV has is a broken render, or worse a silent
                  NA that reads as a real measurement.
  data -> paper   a measured field nothing references is something computed and then ignored,
                  which is how a claim ends up in the prose with no number behind it.

  python scripts/check_paper_data.py            # report both, exit 1 on a missing column
  python scripts/check_paper_data.py --quiet    # only the failures
"""
import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QMD = REPO / "paper" / "separability.qmd"
DATA = REPO / "data"

sys.path.insert(0, str(REPO / "scripts"))
from separability import REPORT_FIELDS  # noqa: E402

# Names that look like columns but are R, ours, or the language's.
IGNORE = {
    "read_csv", "show_col_types", "file_path", "big_mark", "na_rm", "as_tibble", "drop_na",
    "geom_line", "geom_point", "geom_hline", "geom_text", "scale_x_continuous", "fig_env",
    "scale_y_continuous", "scale_color_manual", "scale_fill_manual", "theme_minimal",
    "element_blank", "element_text", "element_line", "panel_grid", "axis_text", "legend_position",
    "plot_title", "strip_text", "labs", "facet_wrap", "kable_styling", "latex_options",
    "repeat_header", "font_size", "booktabs", "longtable", "linesep", "col_names", "escape",
    "group_by", "ungroup", "summarize", "transmute", "arrange", "filter", "mutate", "rename",
    "left_join", "bind_rows", "case_when", "if_else", "n_distinct", "str_detect", "str_replace",
    "toy_cap", "toy_noise", "toy_track", "toy_null", "llm_deep", "grid_stats", "dec_tbl",
    "check_emits", "report_fields", "tbl_cap", "fig_cap", "fig_width", "fig_height",
    "out_width", "fig_align", "warning", "message", "include", "echo", "eval",
}


def qmd_body(text):
    chunks = re.findall(r"```\{r.*?\n(.*?)```", text, re.S)
    inline = re.findall(r"`r ([^`]*)`", text)
    body = "\n".join(chunks + inline)
    return re.sub(r"#.*", "", body)                       # comments name columns they do not read


def gsf_fallbacks(body):
    """Columns the paper reads only through gsf(), the soft getter whose second argument is a
    typed fallback. Those are deliberately optional -- the paper renders today from the fallback
    and switches to the data on the next stats regeneration -- so they are not broken reads."""
    return set(re.findall(r'gsf\(\s*"[^"]*"\s*,\s*"([a-z][a-z0-9_]+)"', body))


def qmd_tokens(body):
    """snake_case identifiers in the R chunks, which is where columns are named."""
    return {t for t in re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", body)} - IGNORE


def qmd_defined(body):
    """Names the paper creates rather than reads: transmute/mutate/summarize targets and
    assignments. `between_adj = std_between_share_adj` renames a column, so `between_adj` is not
    something any CSV is expected to have."""
    return set(re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\s*(?:<-|=(?!=))", body))


def csv_columns():
    cols = {}
    for f in sorted(DATA.glob("*.csv")):
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                hdr = next(csv.reader(fh))
        except (StopIteration, OSError):
            continue
        cols[f.name] = set(hdr)
    return cols


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true", help="only report failures")
    args = ap.parse_args()

    if not QMD.exists():
        print(f"missing {QMD}", file=sys.stderr)
        return 1
    per_file = csv_columns()
    if not per_file:
        print(f"no CSVs under {DATA}; nothing to check against", file=sys.stderr)
        return 1
    every = set().union(*per_file.values())
    body = qmd_body(QMD.read_text(encoding="utf-8"))
    used = qmd_tokens(body)
    made = qmd_defined(body)

    # paper -> data. Only flag tokens that look like data columns: a name the paper uses that is
    # not in any CSV but IS a measured field, or shares a stem with one, is the dangerous kind.
    stems = {c.split("_")[0] for c in every}
    optional = gsf_fallbacks(body)
    suspicious = sorted(t for t in used
                        if t not in every and t not in made and t not in optional
                        and t.split("_")[0] in stems)
    # data -> paper, restricted to what the measure computes, since that is what costs a run.
    prefixed = {p + f for f in REPORT_FIELDS for p in ("", "std_", "raw_")}
    present = sorted(f for f in prefixed if f in every)
    unused = sorted(f for f in present
                    if f not in used and f.replace("std_", "").replace("raw_", "") not in used)

    if not args.quiet:
        print(f"CSVs: {len(per_file)}   distinct columns: {len(every)}")
        print(f"column-like names the paper reads: {len(used)}\n")

    bad = False
    if suspicious:
        bad = True
        print(f"MISSING: {len(suspicious)} name(s) the paper reads that no CSV has")
        for t in suspicious:
            near = sorted(c for c in every if c.startswith(t.split("_")[0]))[:3]
            print(f"    {t:<38} nearest: {', '.join(near) or '-'}")
        print()

    if not args.quiet:
        print(f"MEASURED BUT UNREPORTED: {len(unused)} of {len(present)} measured columns present "
              f"in the data are never read by the paper")
        for f in unused:
            print(f"    {f}")
        print("\n(not an error: it is what has been computed and is available to report)")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
