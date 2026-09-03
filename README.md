# representation-separability

Follow-up to [*Exemplars in Disguise*](../exemplar-abstraction-sims) (Houghton & Kapatsinski).

**Question.** In a distributed representation of a word, is **class-level** structure (what makes
*dog* pattern like *cat* — "noun-ness") a **separable component** of the representation, or is it
entangled with **item-level** structure (what makes *dog* behave like *dog*)? Operationally we
decompose a word-in-context representation `M[item, class]` into a grand mean, an **item marginal**,
a **class marginal**, and an **interaction**, and ask (a) whether those components sit on separate
axes and (b) how large the irreducibly-joint interaction is.

**Headline finding.** *Neither separable nor inseparable — models both abstract and entangle.* The
two marginals are near-orthogonal, yet a large, depth-built item×class **interaction** remains that
cannot be factored back into item plus class, and that interaction is not on axes of its own either:
13–23% of it lies in the span of the two marginals. The balance is **graded across linguistic
dimensions**, most portable for grammatical role, then part of speech, and least for metaphor, which
is carried almost entirely word by word. An ACL methods/results draft is in [`paper/`](paper/).

## Repository structure

Only scripts that produce something in the paper live under `scripts/`. Everything superseded is in
`archive/` (gitignored, kept on disk, and in git history).

```
representation-separability/
├── MATH.md                    # the measure, from plain-language walkthrough to formal reference
├── METHOD.md / NOTES.md       # design decisions and history
├── paper/                     # ACL Quarto draft (separability.qmd + refs.bib)
├── scripts/
│   ├── lib/
│   │   ├── unified_separability.py  # THE measure: decomposition, sizes, overlaps, split-half gate
│   │   └── separability_measure.py  # the older marginal-only `frac`; still imported by llm_extract
│   ├── toy/
│   │   ├── experiment10_abc_grid.py # Experiment 1: the artificial-language grid
│   │   └── validate_measure.py      # Appendix: components planted directly in a representation
│   └── llm/
│       ├── build_concat_ud.py       # concatenate 6 UD English treebanks -> one CoNLL-U
│       ├── llm_extract.py           # extraction, CoNLL-U parsing, label re-derivation
│       ├── run_llm_sweep.py         # drive extraction across the model set (GPU, memmap-streamed)
│       ├── extract_vua.py           # VUA20 metaphor representations (GPU)
│       ├── measure_llm.py           # Experiment 2: part of speech
│       ├── measure_llm_role.py      # Experiment 2: grammatical role
│       ├── measure_llm_metaphor.py  # Experiment 2: metaphor
│       ├── measure_llm_morph.py     # Appendix: number/tense, the not-same-token control
│       ├── decode_from_interaction.py # Experiment 3: decode the class from the interaction
│       ├── methods_stats.py         # grid statistics quoted in the Dataset sections
│       ├── run_ablation.sh          # Appendix: position ablation, end to end, everything to CSV
│       ├── finalize_ablation.sh     # its post-processing stages, runnable on their own
│       └── test_ablation_pipeline.py # regression test for the measurement chain (seconds, no GPU)
├── data/                      # result CSVs are tracked; reps and corpora are gitignored
└── archive/                   # superseded scripts and analyses (gitignored, kept on disk)
```

## Which script produced which result

| paper element | data file | script |
|:--|:--|:--|
| Experiment 1 | `experiment10_abc_grid.csv` | `toy/experiment10_abc_grid.py` |
| Experiment 2, part of speech | `llm_unified_form.csv` | `llm/measure_llm.py --item-key form` |
| Experiment 2, role | `llm_role.csv` | `llm/measure_llm_role.py` |
| Experiment 2, metaphor | `llm_metaphor.csv` | `llm/measure_llm_metaphor.py` |
| Experiment 3 | `llm_decode_pos_form.csv`, `llm_decode_interaction.csv` | `llm/decode_from_interaction.py` |
| Dataset counts | `methods_grid_stats.csv` | `llm/methods_stats.py` |
| Appendix: validation | `validate_measure.csv` | `toy/validate_measure.py` |
| Appendix: morphology | `llm_morph.csv` | `llm/measure_llm_morph.py` |
| Appendix: position ablation | `llm_*_ablation.csv`, `position_ablation_*.csv` | `llm/run_ablation.sh` |

## The measure

`M[item, class] = μ + α(item) + β(class) + γ(item, class)`, a geometric two-way decomposition on
per-dimension-standardized representations (the standardization defends against transformer *rogue
dimensions*). We report the components' **shares** of the between-cell structure and **two
overlaps**: how much of the item marginal lies in the class subspace, and how much of the
interaction lies in the span of the two marginals. Both are read against a chance level of subspace
rank over width, which is measured rather than assumed in the validation appendix.

Because a squared length has a positive noise floor, the interaction is estimated as a **cross
product of two independent half-estimates** of the same grid and tested by permutation. Both
experiments obtain those halves the same way, by splitting a cell's observations, so the toy and the
language models are measured by the same function with the same arguments. See [`MATH.md`](MATH.md).

## Reproducing

**Experiment 1 and the validation appendix** (GPU helps, not required):
```bash
python scripts/toy/experiment10_abc_grid.py     # ~7.5k cells; resumable
python scripts/toy/validate_measure.py          # ~250k runs; resumable, parallel
```

**Experiments 2 and 3.** Extraction needs a GPU; measurement is CPU only. Redirect the read-only HF
cache first (`export HF_HOME=$TMPDIR/hf`):
```bash
python scripts/llm/build_concat_ud.py --out data/ud/en_all-ud.conllu
python scripts/llm/run_llm_sweep.py --conllu data/ud/en_all-ud.conllu \
       --reps-dir data/llm_reps --max-tokens 300000 --device cuda
python scripts/llm/extract_vua.py --out-dir data/vua_reps --device cuda

python scripts/llm/measure_llm.py --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu \
       --item-key form --out data/llm_unified_form.csv
python scripts/llm/measure_llm_role.py  --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu --out data/llm_role.csv
python scripts/llm/measure_llm_morph.py --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu --out data/llm_morph.csv
python scripts/llm/measure_llm_metaphor.py --reps-dir data/vua_reps --out data/llm_metaphor.csv
python scripts/llm/decode_from_interaction.py --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu \
       --out data/llm_decode_interaction.csv
```

**The position-ablation appendix**, one command, everything to CSV:
```bash
nohup setsid bash scripts/llm/run_ablation.sh > logs/ablation.out 2>&1 &
```
It re-extracts the OPT-BabyLM models with `decoder.embed_positions` zeroed (Pythia has no such
module), verifies per model that the ablation applied, measures all three constructions, and writes
the summary. It deletes nothing and is resumable: re-running skips finished extractions.

`python scripts/llm/test_ablation_pipeline.py` checks that whole chain on synthetic data in a few
seconds without a GPU. Worth running before spending hours on an extraction.

## Notes

- **Data policy:** result CSVs under `data/` are versioned; representation directories
  (`data/*_reps/`) and corpora (`data/ud/`) are gitignored, being large and regenerable.
- **Item keying:** every construction in the paper keys the item on the **surface form**, so the
  token is identical at both levels of a distinction. Keying part of speech on the lemma instead
  pools inflected forms and inflates the layer-0 interaction by 50–100×, which is a fact about
  tokenization rather than representation. `measure_llm.py` defaults accordingly.
- **Reps carry their extraction parameters.** Token order depends on `batch_size`, `seed` and
  `max_length`, so those are saved in each `.npz` and used when labels are re-derived. Files written
  before that change are handled by trying the batch sizes this project has used and keeping the one
  that reproduces the saved labels exactly.
- **Paper:** `paper/separability.qmd` uses the [ACL Quarto template](https://github.com/znhoughton/quarto-templates)
  (`quarto add znhoughton/quarto-templates/acl` then `quarto render`).
