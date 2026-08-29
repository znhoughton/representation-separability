# representation-separability

Follow-up to [*Exemplars in Disguise*](../exemplar-abstraction-sims) (Houghton & Kapatsinski).

**Question.** In a distributed representation of a word, is **class-level** structure (what makes
*dog* pattern like *cat* — "noun-ness") a **separable component** of the representation, or is it
entangled with **item-level** structure (what makes *dog* behave like *dog*)? Operationally we
decompose a word-in-context representation `M[item, class]` into a grand mean, an **item marginal**,
a **class marginal**, and an **interaction**, and ask (a) whether the marginals sit on separate axes
(orthogonality) and (b) how large the irreducibly-joint interaction is.

**Headline finding.** *Neither separable nor inseparable — models both abstract and entangle.* The
class marginal is cleanly separable from the item marginal everywhere, yet a significant,
depth-built item×class **interaction** remains that cannot be factored back into class + item. The
balance is **graded across linguistic dimensions**: strong reusable abstraction for part-of-speech,
partial for grammatical role, and *absent* for metaphor (which lives only as a per-lexeme
interaction). Full results in [`SEPARABILITY_FINDINGS.md`](SEPARABILITY_FINDINGS.md); an ACL
methods/results draft in [`paper/`](paper/).

## Repository structure

```
representation-separability/
├── METHOD.md                  # canonical method reference (toy design + measure)
├── SEPARABILITY_FINDINGS.md   # full write-up: toy results (§1–3) + LLM results (§4)
├── NOTES.md                   # design decisions / history
├── paper/                     # ACL Quarto draft (separability.qmd + refs.bib)
├── scripts/
│   ├── lib/                   # shared, imported by everything below
│   │   ├── lowrank_pilot.py       # toy generative task (build_lowrank / build_lowrank_frac)
│   │   ├── toy_models.py          # ModelA_MLP (free per-lexeme embed → MLP) + early-stopping trainer
│   │   ├── separability_measure.py# the `frac` marginal-separability measure
│   │   └── unified_separability.py# unified measure: marginals + interaction (size, orthogonality, gate)
│   ├── toy/                   # controlled toy experiments (motivate + validate the measure)
│   │   ├── experiment8_interaction_grid.py   # frac dose-response grid  (→ §2)   [+ experiment8_gpu, merge_shards]
│   │   ├── experiment9_unified_grid.py       # unified-measure grid      (→ §2.7)
│   │   ├── check_exact_vs_sampled.py         # exact- vs sampled-loss training equivalence
│   │   └── validate_{separability,unified,gt_interaction,superposition}.py
│   └── llm/                   # the payoff: real LM representations
│       ├── build_concat_ud.py     # concat 6 UD English treebanks (+ feats, deprel) → one CoNLL-U
│       ├── run_llm_sweep.py       # extract per-token reps for the model set (GPU, memmap-streamed)
│       ├── llm_extract.py         # extraction + CoNLL-U parsing + label re-derivation
│       ├── measure_llm.py         # POS (noun/verb, noun/adj)
│       ├── measure_llm_role.py    # grammatical role (nsubj/obj)
│       ├── extract_vua.py         # VUA20 metaphor reps (GPU)  →  measure_llm_metaphor.py
│       └── measure_llm_morph.py   # number/tense (multi-token control)
├── data/                      # result CSVs are tracked; reps + corpora are gitignored (regenerable)
└── archive/                   # superseded scripts/analyses/data (gitignored, kept on disk)
```

## The measure

`M[item, class] = μ + α(item) + β(class) + γ(interaction)` (a geometric two-way decomposition on
per-dimension-standardized reps — the standardization defends against transformer *rogue
dimensions*). We report **`size_interaction`** (the interaction's share of the between-cell
structure) and **`leak_item→class`** (how much the item marginal lies in the class subspace;
≈0 = separate axes). Significance and denoising come from two independent estimates + a permutation
null (see [`METHOD.md`](METHOD.md) and [`NOTES.md`](NOTES.md)). The measure is orthogonal-gauge
invariant only (no whitening), so it applies identically to the toy and the LLMs.

## Reproducing

**Toy** (CPU/GPU):
```bash
python scripts/toy/experiment8_interaction_grid.py      # frac grid (§2); experiment8_gpu.py on GPU
python scripts/toy/experiment9_unified_grid.py          # unified-measure grid (§2.7)
python scripts/toy/validate_unified.py                  # measure validation suite
```

**LLM** (extraction needs a GPU; measurement is CPU). Redirect the read-only HF cache first
(`export HF_HOME=$TMPDIR/hf`):
```bash
python scripts/llm/build_concat_ud.py --out data/ud/en_all-ud.conllu           # corpus
python scripts/llm/run_llm_sweep.py  --conllu data/ud/en_all-ud.conllu \
       --reps-dir data/llm_reps --device cuda                                   # reps (GPU)
python scripts/llm/measure_llm.py       --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu --out data/llm_unified.csv
python scripts/llm/measure_llm_role.py  --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu --out data/llm_role.csv
python scripts/llm/measure_llm_morph.py --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu --out data/llm_morph.csv
python scripts/llm/extract_vua.py --out-dir data/vua_reps --device cuda         # metaphor reps (GPU)
python scripts/llm/measure_llm_metaphor.py --reps-dir data/vua_reps --out data/llm_metaphor.csv
```

## Notes

- **Data policy:** result CSVs under `data/` are versioned; representation dirs (`data/*_reps/`) and
  UD/VUA corpora (`data/ud/`) are gitignored — they are large and regenerable from the scripts above.
- **Environment:** a shared, GPU-equipped box; extraction is memory-streamed to stay well under the
  RAM cap. See `NOTES.md` for environment specifics.
- **Paper:** `paper/separability.qmd` uses the [ACL Quarto template](https://github.com/znhoughton/quarto-templates)
  (`quarto add znhoughton/quarto-templates/acl` then `quarto render`).
