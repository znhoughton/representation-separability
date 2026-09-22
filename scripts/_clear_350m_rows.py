"""Drop rows for one model id from the paper's result CSVs.

Used by rerun_350m_prenorm.sh. The corrected 350M carries the same HuggingFace
name as the model it replaces, so run_all_measurements.sh would treat the old
rows as already-done work and skip the model entirely. Clearing them first is
what makes the re-run actually happen.
"""
import csv, glob, os, sys

mid = sys.argv[1]
for path in sorted(glob.glob("data/llm_*.csv")):
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            rows = list(rd)
            fields = rd.fieldnames
    except Exception as err:
        print(f"  {path}: unreadable ({err})")
        continue
    if not rows or not fields or "model" not in fields:
        continue
    n = sum(r["model"] == mid for r in rows)
    if not n:
        print(f"  {path}: no 350M rows")
        continue
    kept = [r for r in rows if r["model"] != mid]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    os.replace(tmp, path)
    print(f"  {path}: cleared {n} old 350M rows")
