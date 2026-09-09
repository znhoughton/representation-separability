"""Repair a results CSV in place before anything resumes from it.

Two things go wrong when a run is killed. The last row can be half written, and the resume logic
skips rows it cannot parse, so that cell is measured again and appended a second time. Both are
silent: the file still loads, and the duplicate wins or loses depending on which the reader keeps.

Called at startup by the scripts that resume, so a killed run needs no manual cleanup.
"""
import csv
import os


def repair(path, key_cols=None, verbose=True):
    """Drop an incomplete final row and any duplicate keys, keeping the last of each.

    `key_cols` identifies a row for duplicate detection; without it only truncation is fixed.
    Returns (rows_dropped_truncated, rows_dropped_duplicate).
    """
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0, 0

    with open(path, newline="", encoding="utf-8") as fh:
        raw = fh.read()
    if not raw.strip():
        return 0, 0

    # A file not ending in a newline had its last line cut off mid-write.
    torn = 0
    if not raw.endswith("\n"):
        cut = raw.rfind("\n")
        raw = raw[:cut + 1] if cut != -1 else ""
        torn = 1

    rows = list(csv.reader(raw.splitlines()))
    if not rows:
        return torn, 0
    header, body = rows[0], rows[1:]

    # A row with the wrong field count is also a partial write, even if a newline followed it.
    good = [r for r in body if len(r) == len(header)]
    torn += len(body) - len(good)

    dupes = 0
    if key_cols and all(c in header for c in key_cols):
        idx = [header.index(c) for c in key_cols]
        seen, keep = {}, []
        for r in good:                       # last occurrence wins: it is the completed rerun
            seen[tuple(r[i] for i in idx)] = r
        for r in good:
            k = tuple(r[i] for i in idx)
            if seen.get(k) is r:
                keep.append(r)
        dupes = len(good) - len(keep)
        good = keep

    if torn or dupes:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(good)
        if verbose:
            bits = []
            if torn:
                bits.append(f"{torn} truncated")
            if dupes:
                bits.append(f"{dupes} duplicate")
            print(f"  repaired {path}: dropped {' and '.join(bits)} row(s)", flush=True)
    return torn, dupes
