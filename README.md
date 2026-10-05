# representation-separability

Follow-up to *Exemplars in Disguise*.

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
dimensions**, more portable for grammatical role than for part of speech. The rendered paper is in [`paper/`](paper/); its Quarto source is withheld
while the work is under double-blind review.

## Repository structure

Only scripts that produce something in the paper live under `scripts/`.

```
representation-separability/
├── paper/                          # rendered PDF and refs.bib (source withheld during review)
├── scripts/
│   ├── separability.py             # THE measure (numpy): decomposition, sizes, overlaps, 200-re-split CI
│   ├── separability_gpu.py         # torch per-spec backend (LLM + toy re-measure); SEP_DEVICE=cuda
│   ├── separability_batch.py       # torch batched backend (validation): same-shape specs in one batch
│   ├── verify_backends.py          # checks the two GPU backends match numpy, component by component
│   ├── run_all_measurements.sh     # every measurement in one command; resumable, deletes nothing
│   ├── csv_repair.py               # repairs a killed run's CSV and migrates a stale header
│   ├── check_paper_data.py         # columns the paper reads vs the data; and what goes unreported
│   ├── check_paper_renders.R       # runs the paper's R against the data, no LaTeX needed
│   ├── check_between_bias.py       # why the between-grid share is a variance component, not raw
│   ├── dev/                        # developer tooling, not part of the pipeline (GPU benchmarks)
│   ├── toy/
│   │   ├── artificial_language_grid.py  # Experiment 1
│   │   ├── remeasure_from_runs.py       # re-measures the grid from saved states, no retraining
│   │   └── validate_measure.py          # Appendix: components planted directly in a representation
│   └── llm/
│       ├── build_ud_corpus.py           # concatenate 6 UD English treebanks -> one CoNLL-U
│       ├── extraction.py                # the extraction library: forward passes, alignment, labels
│       ├── extract_ud.py                # drive extraction over UD for the model set (GPU)
│       ├── extract_vua.py               # the same for VUA20 metaphor (GPU; not used by the paper)
│       ├── measure.py                    # constructions: pos, role, morphology (also metaphor, unused by the paper)
│       ├── decode_from_interaction.py   # Experiment 3
│       ├── dataset_stats.py             # the counts quoted in the Dataset/Stimuli sections,
│       │                                #   plus the per-item stimulus lists (stimuli_items.csv)
│       ├── run_position_ablation.sh     # Appendix: the ablation end to end, everything to CSV
│       ├── finalize_position_ablation.sh  # its post-processing, runnable on its own
│       └── test_measurement_pipeline.py # regression test for the chain (seconds, no GPU)
├── data/                           # result CSVs are tracked; reps and corpora are gitignored
├── notes/                          # drawio sources for the appendix figures (exports gitignored)
├── old/                            # CSVs a rerun superseded, timestamped; gitignored, kept on disk
└── archive/                        # earlier versions, gitignored (see archive/README.md)
```

## Which script produced which result

All of these are produced by `scripts/run_all_measurements.sh`; the third column is what to run
if you want just one of them.

| paper element | data file | script |
|:--|:--|:--|
| Experiment 1 | `artificial_language_grid.csv` | `toy/artificial_language_grid.py` |
| Experiment 2, part of speech | `llm_unified_form.csv` | `llm/measure.py pos --item-key form` |
| Experiment 2, role | `llm_role.csv` | `llm/measure.py role` |
| Experiment 3 | `llm_decode_pos_form.csv`, `llm_decode_interaction.csv` | `llm/decode_from_interaction.py` |
| Dataset counts | `methods_grid_stats.csv`, `stimuli_items.csv` | `llm/dataset_stats.py` |
| Appendix: validation | `validate_measure.csv.gz` (uncompressed is gitignored) | `toy/validate_measure.py` |
| Appendix: morphology | `llm_morph.csv` | `llm/measure.py morphology` |
| Appendix: position ablation | `llm_*_ablation.csv`, `position_ablation_*.csv` | `llm/run_position_ablation.sh` |

## The measure

`M[item, class] = μ + α(item) + β(class) + γ(item, class)`, a geometric two-way decomposition on
per-dimension-standardized representations (the standardization defends against transformer *rogue
dimensions*). We report the components' **shares** of the between-cell structure and **two
overlaps**: how much of the item marginal lies in the class subspace, and how much of the
interaction lies in the span of the two marginals. Both are read against a chance level of subspace
rank over width, which is measured rather than assumed in the validation appendix.

Because a squared length has a positive noise floor, **each of the three sizes** (item, class and
interaction) is estimated as a **cross product of two independent half-estimates** of the same grid,
whose null is exactly zero. Significance is a **200-draw re-split confidence interval**: the split of
each cell's observations is redrawn 200 times and a size counts as present when that interval
excludes zero, calibrated to a ~5% false-positive rate on planted zeros. The two overlaps are read
against a random-orientation null instead. Both experiments obtain the halves the same way, so the
toy and the language models are measured by the same function with the same arguments. The paper's
first appendix gives the formal statement.

The measure runs identically on CPU (numpy) or GPU (torch): set `SEP_DEVICE=cuda` and every
measurement routes through the torch backends (`separability_gpu.py` per-spec, `separability_batch.py`
batched for the validation grid), which `verify_backends.py` checks against numpy component by
component. The default is CPU.

## Reproducing

Extract the representations once, then run everything else with one command:

```bash
# once: build the corpus and extract. Needs a GPU and downloads the models.
export HF_HOME=$TMPDIR/hf                       # the default cache is often read-only
python scripts/llm/build_ud_corpus.py --out data/ud/en_all-ud.conllu
python scripts/llm/extract_ud.py  --conllu data/ud/en_all-ud.conllu --device cuda
python scripts/llm/extract_vua.py --device cuda

# everything the paper reads, in dependency order
mkdir -p logs
nohup setsid bash scripts/run_all_measurements.sh > logs/all.out 2>&1 &
tail -f logs/all.out
```

That produces all ten data files, runs Experiment 1 on the GPU, and finishes by reporting how many
values landed in each column and whether every construction covers all six models. It is resumable
and deletes nothing: a superseded file is moved to `old/` with a timestamp. `ABLATION=1` adds the
position-ablation appendix, which is off by default because it re-extracts.

It also writes two directories of artefacts, both gitignored:

| | what | size |
|:--|:--|:--|
| `data/toy_runs/` | per toy cell: the hidden states the measure ran on, and the raw null draws | ~8 GB |
| `data/llm_nulls/` | per model and construction: the raw null draws | a few hundred KB |

These exist so that a change to how a number is *summarised* never costs another measurement pass.
Twice it has, and once it cost retraining the whole toy grid, because only one quantile of a null
distribution had been written down. `--no-save-runs` skips the 8 GB if you do not want it.

### Running one piece

**Experiment 1 and the validation appendix** (GPU helps, not required):
```bash
python scripts/toy/artificial_language_grid.py     # ~7.5k cells; resumable
python scripts/toy/validate_measure.py          # ~250k runs; resumable, parallel
```

**Experiments 2 and 3.** Extraction needs a GPU; measurement runs on CPU by default, or on GPU with
`SEP_DEVICE=cuda`:
```bash
python scripts/llm/measure.py pos --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu \
       --item-key form --out data/llm_unified_form.csv
python scripts/llm/measure.py role  --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu --out data/llm_role.csv
python scripts/llm/measure.py morphology --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu --out data/llm_morph.csv
python scripts/llm/decode_from_interaction.py --reps-dir data/llm_reps --conllu data/ud/en_all-ud.conllu \
       --out data/llm_decode_interaction.csv
```

**The position-ablation appendix**, one command, everything to CSV:
```bash
nohup setsid bash scripts/llm/run_position_ablation.sh > logs/ablation.out 2>&1 &
```
It re-extracts the OPT-BabyLM models with `decoder.embed_positions` zeroed (Pythia has no such
module), verifies per model that the ablation applied, measures all three constructions, and writes
the summary. It deletes nothing and is resumable: re-running skips finished extractions.

`python scripts/llm/test_measurement_pipeline.py` checks that whole chain on synthetic data in a few
seconds without a GPU. Worth running before spending hours on an extraction.

## Notes

- **Stimuli:** the words the Experiment 2 grids are built from are listed in
  `data/stimuli_items.csv`, one row per kept item per model, with its two class labels and the
  token count behind each grid cell of it (`pos_noun_verb_form`: 78 noun/verb conversion forms
  like *run*, *hit*, *claims*; `role_nsubj_obj`: 49 UD nouns attested as both subject and
  object). Both are same-token, keyed on the lowercased surface form, min 10 tokens
  per class. Regenerate with `python scripts/llm/dataset_stats.py` on the box holding the reps;
  the paper's Stimuli section (@sec-exp2-stimuli) summarizes the same file.
- **Data policy:** result CSVs under `data/` are versioned, the one exception being the
  validation grid, which is versioned compressed (`validate_measure.csv.gz`; its uncompressed
  working copy is gitignored). Representation directories (`data/*_reps/`) and corpora
  (`data/ud/`) are gitignored, being large and regenerable.
- **Item keying:** every construction in the paper keys the item on the **surface form**, so the
  token is identical at both levels of a distinction. Keying part of speech on the lemma instead
  pools inflected forms and inflates the layer-0 interaction by 50–100×, which is a fact about
  tokenization rather than representation. `measure_pos.py` defaults accordingly.
- **Reps carry their extraction parameters.** Token order depends on `batch_size`, `seed` and
  `max_length`, so those are saved in each `.npz` and used when labels are re-derived. Files written
  before that change are handled by trying the batch sizes this project has used and keeping the one
  that reproduces the saved labels exactly.
- **Figures:** the two diagrams are drawio sources; Quarto cannot read those, so `bash paper/make-figures.sh` exports the PDFs the qmd includes. Re-run it after editing a diagram.
- **Paper:** the Quarto source is not included while the work is under review. It builds
  with a custom ACL template (`quarto add <template-repo>/acl`, then `quarto render`).
