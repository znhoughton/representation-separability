"""Join the context-only control against the matched word baseline and emit a comparison.

For each construction (part of speech, grammatical role) this reads the two CSVs
run_prev_context_control.sh produced -- the control (*_prevtok) and the word baseline measured on
the identical dropped-initial token set (*_worddrop) -- lines them up on (model, layer), and reports
the interaction side by side.

Read it as magnitude AND geometry, not the share alone: before the word is seen the item marginal is
weak, so the interaction's SHARE of a smaller pie can look large even when the word-token interaction
is bigger in absolute terms. We therefore print size_interaction (share) next to size_item and the
interaction->margins leak, and the control "passes" where the word-token interaction is both larger
and grows with depth while the context-only one stays flat/low.

  python scripts/llm/summarize_prev_context.py --dir data --out data/prev_context_control_summary.csv
"""
import argparse
from pathlib import Path

import pandas as pd

# (construction, control csv, baseline csv, column prefix). pos is written with a std_/raw_ prefix
# (we read the standardized measure); role has no prefix.
CONSTRUCTIONS = [
    ("pos",  "llm_unified_form_prevtok.csv", "llm_unified_form_worddrop.csv", "std_"),
    ("role", "llm_role_prevtok.csv",         "llm_role_worddrop.csv",         ""),
]
# the fields we line up; stored under these bare names regardless of the csv's prefix.
# between_share is the fraction of the whole representation that is between-cell structure (the rest
# is context/within-cell), so size_interaction * between_share is the interaction as a fraction of
# the WHOLE token state -- the fair word-vs-prev magnitude (size_interaction alone is a share of a
# pie whose size differs between the two).
FIELDS = ["size_interaction", "size_item", "size_class", "leak_int_into_margins",
          "between_share", "obs_size_interaction", "size_interaction_excludes_zero"]


def _load(path, prefix, init):
    df = pd.read_csv(path)
    if "init" in df.columns:
        df = df[df["init"].astype(str) == init]
        if df.empty:
            raise SystemExit(f"{path}: no rows with init=={init!r}")
    keep = {"model": "model", "layer": "layer"}
    keep.update({prefix + f: f for f in FIELDS})
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    out = df[list(keep)].rename(columns=keep)
    # interaction as a fraction of the whole representation (divides out the pie-size difference)
    out["int_of_whole"] = out["size_interaction"] * out["between_share"]
    # relative depth within each model: 0 = embeddings, 1 = final block
    out["reldepth"] = out.groupby("model")["layer"].transform(lambda s: s / s.max() if s.max() else 0.0)
    return out


def _short(model):
    return model.split("/")[-1].replace("-20eps-seed964", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data", help="directory holding the four CSVs")
    ap.add_argument("--out", default="data/prev_context_control_summary.csv")
    ap.add_argument("--init", default="pretrained",
                    help="which init condition to compare (default pretrained; the reps dir may also "
                         "hold *_noposemb ablation files, which this filters out)")
    args = ap.parse_args()
    d = Path(args.dir)

    tidy = []
    for con, cfile, wfile, prefix in CONSTRUCTIONS:
        cp, wp = d / cfile, d / wfile
        if not cp.exists() or not wp.exists():
            print(f"[skip {con}] missing {cp.name if not cp.exists() else wp.name}")
            continue
        ctrl = _load(cp, prefix, args.init).add_prefix("prev_")
        base = _load(wp, prefix, args.init).add_prefix("word_")
        m = base.merge(ctrl, left_on=["word_model", "word_layer"], right_on=["prev_model", "prev_layer"])
        m = m.rename(columns={"word_model": "model", "word_layer": "layer", "word_reldepth": "reldepth"})
        m["construction"] = con
        tidy.append(m)

        print(f"\n================  {con.upper()}  (init={args.init})  ================")
        print("  int_of_whole = interaction as a fraction of the WHOLE state (the fair comparison);")
        print("  share = interaction's share of between-cell structure (pie differs word vs prev).")
        for model, g in m.sort_values(["model", "layer"]).groupby("model"):
            print(f"\n  {_short(model)}")
            print(f"    {'layer':>5} {'depth':>5} | {'word_whole':>10} {'prev_whole':>10} | "
                  f"{'word_share':>10} {'prev_share':>10} | {'word_btwn':>9} {'prev_btwn':>9}")
            for _, r in g.iterrows():
                print(f"    {int(r['layer']):>5} {r['reldepth']:>5.2f} | "
                      f"{r['word_int_of_whole']:>10.4f} {r['prev_int_of_whole']:>10.4f} | "
                      f"{r['word_size_interaction']:>10.3f} {r['prev_size_interaction']:>10.3f} | "
                      f"{r['word_between_share']:>9.3f} {r['prev_between_share']:>9.3f}")
            deep = g.loc[g["layer"].idxmax()]
            print(f"    deepest (L{int(deep['layer'])}): interaction-of-whole "
                  f"word {deep['word_int_of_whole']:.4f} vs context-only {deep['prev_int_of_whole']:.4f}")

    if not tidy:
        raise SystemExit("no constructions found; run run_prev_context_control.sh first")
    cols = ["construction", "model", "layer", "reldepth",
            "word_int_of_whole", "prev_int_of_whole",
            "word_size_interaction", "prev_size_interaction",
            "word_between_share", "prev_between_share",
            "word_size_item", "prev_size_item", "word_size_class", "prev_size_class",
            "word_leak_int_into_margins", "prev_leak_int_into_margins",
            "word_obs_size_interaction", "prev_obs_size_interaction",
            "word_size_interaction_excludes_zero", "prev_size_interaction_excludes_zero"]
    out = pd.concat(tidy, ignore_index=True)[cols]
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
