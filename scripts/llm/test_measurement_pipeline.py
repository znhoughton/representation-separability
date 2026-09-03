"""End-to-end check of the ablation measurement chain on synthetic reps.

Reproduces the condition that lost the last run: a reps directory holding BOTH the intact and the
_noposemb file for each model. Verifies that the ablated files are measured rather than skipped,
that an `init` column distinguishes them, and that finalize_position_ablation.sh then builds a 2x2 with a
`zeroed` row.

Three routes, chosen by what each script depends on:
  metaphor  -- measure_file needs no tokenizer, so main() runs for real, worker pool included.
  POS/role  -- measure_file re-derives labels with the model's tokenizer, which would download
               models. Those two are checked in two parts instead: measure_file is called
               in-process with the derivation stubbed (proving it records init), and main() is
               run far enough to print its todo count (proving the resume check no longer treats
               the ablated file as already done -- the actual bug).
"""
import contextlib
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(r"D:\PhD Stuff\Linguistics Stuff\representation-separability")
SCRATCH = Path(tempfile.gettempdir()) / "abltest"
for sub in ("lib", "llm", "toy"):
    sys.path.insert(0, str(REPO / "scripts" / sub))

RNG = np.random.default_rng(964)
N_ITEM, N_PER_CELL, D = 12, 8, 16
LAYERS = [0, 1]
MODELS = ["znhoughton/opt-babylm-125m-20eps-seed964", "znhoughton/opt-babylm-350m-20eps-seed964"]
INITS = ["pretrained", "pretrained_noposemb", "random", "random_noposemb"]
FAILURES = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(label)


def make_ud_npz(path, model, init):
    upos, lemma, form, deprel = [], [], [], []
    for i in range(N_ITEM):
        for cls in ("NOUN", "VERB"):
            for k in range(N_PER_CELL):
                upos.append(cls); lemma.append(f"item{i}"); form.append(f"item{i}")
                deprel.append(("nsubj" if k < N_PER_CELL // 2 else "obj") if cls == "NOUN" else "root")
    n = len(upos)
    arrs = {f"layer_{li}": RNG.normal(size=(n, D)).astype(np.float32) for li in LAYERS}
    np.savez_compressed(path, upos=np.array(upos), lemma=np.array(lemma),
                        layer_idxs=np.array(LAYERS), model=model, init=init, **arrs)
    return np.array(upos), np.array(lemma), np.array(form), np.array(deprel)


def make_vua_npz(path, model, init):
    form, label = [], []
    for i in range(N_ITEM):
        for lab in (0, 1):
            for _ in range(N_PER_CELL):
                form.append(f"word{i}"); label.append(lab)
    n = len(form)
    arrs = {f"layer_{li}": RNG.normal(size=(n, D)).astype(np.float32) for li in LAYERS}
    np.savez_compressed(path, form=np.array(form), pos=np.array(["VERB"] * n),
                        label=np.array(label, dtype=np.int64), layer_idxs=np.array(LAYERS),
                        model=model, init=init, **arrs)


def build(reps, vua):
    labels = None
    for m in MODELS:
        for init in INITS:
            stem = m.replace("/", "__")
            labels = make_ud_npz(reps / f"{stem}__{init}.npz", m, init)
            if init.startswith("pretrained"):
                make_vua_npz(vua / f"{stem}__{init}.npz", m, init)
    return labels


def main():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    reps, vua, data = SCRATCH / "reps", SCRATCH / "vua", SCRATCH / "data"
    for d in (reps, vua, data):
        d.mkdir(parents=True)
    upos, lemma, form, deprel = build(reps, vua)
    os.chdir(SCRATCH)

    n_ablated_files = len(list(reps.glob("*_noposemb.npz")))
    print(f"\nsynthetic reps: {len(list(reps.glob('*.npz')))} UD files "
          f"({n_ablated_files} ablated), {len(list(vua.glob('*.npz')))} VUA files\n")

    # ---------------------------------------------------------------- metaphor, for real
    print("metaphor (full main(), real worker pool):")
    import measure
    out = data / "met.csv"
    sys.argv = ["measure", "metaphor", "--reps-dir", str(vua), "--min-cell", "2",
                "--workers", "2", "--out", str(out)]
    with contextlib.redirect_stdout(io.StringIO()):
        measure.main()
    rows = list(csv.DictReader(open(out)))
    inits = sorted({r.get("init", "<MISSING>") for r in rows})
    check(any("noposemb" in i for i in inits), "ablated rows present", ",".join(inits))
    check(len({(r["model"], r["init"]) for r in rows}) == 4, "all 4 (model,init) measured",
          f"{len({(r['model'], r['init']) for r in rows})} combos")

    # resume must not re-measure, and must not skip a condition it has not seen
    before = len(rows)
    with contextlib.redirect_stdout(io.StringIO()):
        measure.main()
    check(len(list(csv.DictReader(open(out)))) == before, "re-run adds nothing (resume works)")

    # ------------------------------------------------- POS and role: measure_file records init
    print("\nPOS / role (measure_file in-process, label derivation stubbed):")
    import extraction
    stub = {"upos": upos, "lemma": lemma, "form": form, "deprel": deprel}
    extraction.derive_labels = lambda *a, **k: stub
    extraction.parse_conllu = lambda *a, **k: []

    import measure
    measure.derive_labels = extraction.derive_labels
    measure.parse_conllu = extraction.parse_conllu

    f_abl = reps / f"{MODELS[0].replace('/', '__')}__pretrained_noposemb.npz"
    import types
    A = types.SimpleNamespace(min_cell=2, conllu="x", item_key="form", classes="NOUN,VERB")
    r_pos = measure.measure_pos(str(f_abl), A, None)
    check(bool(r_pos) and all(r["init"] == "pretrained_noposemb" for r in r_pos),
          "measure.measure_pos tags init", f"{len(r_pos)} rows")
    r_role = measure.measure_role(str(f_abl), A, None)
    check(bool(r_role) and all(r["init"] == "pretrained_noposemb" for r in r_role),
          "measure.measure_role tags init", f"{len(r_role)} rows")

    # ------------------------------------------- POS and role: the resume check (the real bug)
    print("\nPOS / role (resume check with intact rows already in the CSV):")
    for name, constr, argv_extra, outname in [
        ("POS", "pos", ["--item-key", "form", "--conllu", "x"], "form.csv"),
        ("role", "role", ["--conllu", "x"], "role.csv"),
    ]:
        out = data / outname
        # Seed the CSV with ONLY the intact conditions, exactly as the failed run had it.
        fields = getattr(measure, {"pos": "POS_FIELDS", "role": "ROLE_FIELDS"}[constr])
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for m in MODELS:
                for init in ("pretrained", "random"):
                    w.writerow({"model": m, "init": init, "layer": 0})
        sys.argv = (["measure", constr, "--reps-dir", str(reps), "--min-cell", "2",
                     "--workers", "1", "--out", str(out)] + argv_extra)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                measure.main()
        except Exception:
            pass                      # the pool cannot run here; we only need the todo count
        txt = buf.getvalue()
        m_todo = re.search(r"on (\d+) files|measuring (\d+) files", txt)
        todo = int(next(g for g in m_todo.groups() if g)) if m_todo else -1
        skipped = re.findall(r"SKIP (\S+)", txt)
        check(todo == n_ablated_files, f"{name}: ablated files queued, intact skipped",
              f"todo={todo} (want {n_ablated_files}), skipped={len(skipped)}")
        check(all("noposemb" not in s for s in skipped),
              f"{name}: no ablated file was skipped as already-done")

    # --------------------------------------------------------------- finalize builds the 2x2
    print("\nfinalize_position_ablation.sh (summary from the measurement CSVs):")
    shutil.copy(data / "met.csv", data / "llm_metaphor_ablation.csv")
    # a POS csv with both conditions, so the 2x2 has an intact and a zeroed row
    with open(data / "llm_unified_form_ablation.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "init", "layer", "std_size_interaction"])
        for init, val in [("random", 0.088), ("random_noposemb", 0.004),
                          ("pretrained", 0.115), ("pretrained_noposemb", 0.091)]:
            w.writerow([MODELS[0], init, 12, val])
    sh = str(REPO / "scripts" / "llm" / "finalize_position_ablation.sh").replace("\\", "/")
    bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
    rc = subprocess.run([bash, sh, "summary"],
                        capture_output=True, text=True, cwd=str(SCRATCH))
    twox2 = SCRATCH / "data" / "position_ablation_2x2.csv"
    check(twox2.exists(), "2x2 written", rc.stdout.strip().splitlines()[-1] if rc.stdout else rc.stderr[:60])
    if twox2.exists():
        rows = list(csv.DictReader(open(twox2)))
        pos_vals = {r["positions"] for r in rows}
        check("zeroed" in pos_vals, "2x2 contains a `zeroed` row", ",".join(sorted(pos_vals)))
        bl = [r for r in rows if r["trained"] == "no" and r["positions"] == "zeroed"]
        check(bool(bl), "the decisive cell (untrained + zeroed) is present",
              f"interaction={bl[0]['interaction']}" if bl else "MISSING")

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
