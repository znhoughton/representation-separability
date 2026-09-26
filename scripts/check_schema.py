#!/usr/bin/env python
"""Every CSV writer must cover every field the measure emits, and carry no field it does not.

run_all_measurements.sh has long checked the first half of that (a writer missing a column, which
is only noticed when someone wants the number and by then costs another full pass). It never
checked the second half, and that is the half that bit: removing sig_item / sig_class /
sig_interaction from the measure left them in the writers' field lists, where they would have been
emitted as empty columns on every row rather than failing loudly.

  python scripts/check_schema.py        # exits non-zero and names the offender
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for sub in ("", "toy", "llm"):
    sys.path.insert(0, str(REPO / "scripts" / sub))

from separability import REPORT_FIELDS, check_emits          # noqa: E402
import measure as M                                          # noqa: E402
from artificial_language_grid import FIELDS as TOY           # noqa: E402
from validate_measure import FIELDS as VAL                   # noqa: E402

# Columns the measure used to emit and no longer does. A writer still listing one would write it
# empty on every row, which reads as "measured and missing" rather than "not measured".
RETIRED = ("sig_item", "sig_class", "sig_interaction")

WRITERS = (
    (M.POS_FIELDS,   ("std_", "raw_"), "measure.py pos"),
    (M.ROLE_FIELDS,  ("",),            "measure.py role"),
    (M.MET_FIELDS,   ("",),            "measure.py metaphor"),
    (M.MORPH_FIELDS, ("",),            "measure.py morphology"),
    (TOY,            ("",),            "toy grid"),
    (VAL,            ("",),            "validate_measure"),
)


def main():
    bad = []
    for fields, prefixes, label in WRITERS:
        check_emits(fields, prefixes, label)                 # raises if a field is missing
        stale = [f for f in fields
                 if any(f == p + r for p in prefixes for r in RETIRED)]
        if stale:
            bad.append(f"{label} still lists retired columns: {stale}")
    stale_report = [f for f in REPORT_FIELDS if f in RETIRED]
    if stale_report:
        bad.append(f"REPORT_FIELDS still lists retired columns: {stale_report}")

    if bad:
        for line in bad:
            print(f"  [schema] {line}", file=sys.stderr)
        return 1
    print(f"  [schema] {len(WRITERS)} writers cover all {len(REPORT_FIELDS)} measured fields, "
          f"none carries a retired column")
    return 0


if __name__ == "__main__":
    sys.exit(main())
