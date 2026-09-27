# AmazonML2026 — Business Entity Resolution Challenge

> **Ownership handoff:** This repository has been handed off to a new team, who now own and
> maintain it going forward. This README is written to get you from zero to a reproduced
> submission without needing to ask the previous team anything.

---

## 1. The problem

This is a solution to the **Business Entity Resolution Challenge**, a 72-hour ML hackathon
(Amazon ML Challenge 2026, hosted on Unstop). Full original problem statement, verbatim: [`PS.md`](PS.md).

**In one sentence:** given business records from three independent, noisy data sources, find
every record in Source 2 and Source 3 that describes the *same real-world business* as each
record in Source 1 — without guessing, because a wrong merge is penalized twice as hard as a
missed one.

**Inputs** (tab-separated, `entity_id` prefixed by source — `S1-`/`S2-`/`S3-`):

| File | Columns | Notes |
|---|---|---|
| `train_source1.tsv` | `entity_id, business_name, business_address, country` | ~2.2M rows. The deduplicated reference source. |
| `train_source2.tsv` / `train_source3.tsv` | same | ~5.0M / ~5.3M rows. Noisy records to match against Source 1. |
| `train_ground_truth.tsv` | `source1_entity_id, matched_entity_ids` (comma-separated) | Used only for training/validation, never available at test time. |
| `test_source{1,2,3}.tsv` | same as train | 1.7M / 4.9M / 5.1M rows. Test additionally contains **France**, a country never seen in training. |

**Required outputs**, both TSV, placed in `output/`:

- **`matching_results.tsv`** — one row per Source-1 test entity, with a comma-separated list of
  matched Source-2/3 IDs (empty if none). **This is the only file that is scored.**
- **`candidate_pairs.tsv`** — the candidate set your blocking stage produced, *before* the final
  model narrowed it down. Not scored, but used to audit blocking quality (recall ceiling,
  reduction ratio) and to verify `matching_results.tsv` is a subset of it.

**Scoring:** macro-averaged **F0.5** (β=0.5), computed per Source-1 entity then averaged.
F0.5 weights precision 2× over recall — a false merge costs more than a missed match. A Source-1
entity with genuinely zero matches (a "singleton") scores 1.0 if you correctly predict an empty
list, and 0.0 if you predict anything for it.

**Hard constraints:**
- No external data, APIs, or lookups anywhere in the pipeline (disqualifying offense) — only the
  provided files and in-code, fixed logic.
- Final model must be **MIT/Apache-2.0 licensed and ≤8B parameters**.
- Output format must exactly satisfy `AmazonML/student_resource/utils/validate_submission.py`
  (run it before every submission — only 5 submissions/day are allowed).

Why this is hard: comparing every Source-1 record against every Source-2/3 record is
~2.2M × 10.3M ≈ 22 trillion pairs — computationally impossible. The whole design problem is how
to narrow that down (**blocking**) without losing true matches, before you can even start
classifying pairs.

---

## 2. System architecture

```
                     ┌────────────────────────────────────────────────────┐
raw *.tsv  ────────▶ │ Stage 0 — Normalize                                 │
(3 sources)          │ transliteration, legal-suffix/honorific stripping,  │
                     │ address abbreviation expansion, number extraction   │
                     └───────────────────────┬────────────────────────────┘
                                              ▼
                     ┌────────────────────────────────────────────────────┐
                     │ Stage 1 — Blocking / Candidate Generation           │
                     │  • GPU semantic search (multilingual-e5-small)     │
                     │  • exact-key join on normalized core name          │
                     │  • exact-key join on name-token + house-number     │
                     │  all restricted to matching country, unioned       │
                     └───────────────────────┬────────────────────────────┘
                                              ▼  candidate_pairs.tsv
                     ┌────────────────────────────────────────────────────┐
                     │ Stage 2 — Pairwise Matching Model                   │
                     │  16 similarity/context features per candidate pair │
                     │  → LightGBM binary classifier → match probability  │
                     └───────────────────────┬────────────────────────────┘
                                              ▼
                     ┌────────────────────────────────────────────────────┐
                     │ Stage 3 — Decoding                                  │
                     │  exclusive per-record argmax above threshold τ     │
                     │  (each S2/S3 record → at most one S1 entity)       │
                     └───────────────────────┬────────────────────────────┘
                                              ▼
                          matching_results.tsv + candidate_pairs.tsv
```

| Stage | Script(s) (in `code/business_entity_resolution/src/`) | What it does |
|---|---|---|
| 0. Normalize | `normalize.py` | Transliterates non-Latin scripts to Latin, lowercases, strips punctuation noise, expands address abbreviations, strips legal suffixes/honorifics to produce a "core" name, extracts house numbers/PIN codes. Cached to `work/norm_*.parquet`. |
| 1. Blocking | `embed.py`, `retrieve.py`, `run_retrieve.py` | Embeds `name \| address` with a GPU sentence-transformer, does chunked top-K cosine retrieval per country, unions with two lexical safety-net joins. Writes `work/candidates_{split}.parquet` and `work/candidate_pairs_{split}.tsv`. |
| 2. Features | `features.py`, `run_features.py` | Computes 16 pairwise features (string similarity, address/number overlap, embedding cosine, retrieval rank/margin, generic-name frequency), out-of-core, parallelized across workers. Writes `work/pairs_features_{split}.parquet`. |
| 2b. Train | `split_and_sample.py`, `train_matcher.py` | Splits train by S1 entity (90/10), trains the LightGBM classifier. Writes `work/lgbm_matcher.txt`. |
| 3. Decode | `decode.py`, `evaluate.py`, `score_matcher.py`, `run_test_finish.py` | Sweeps the decode threshold τ against real macro F0.5 on the held-out split, then scores + decodes the test candidates into the two submission files. |
| — | `config.py` | All paths/constants, derived from the repo layout — nothing to hand-edit. |
| — | `io_utils.py` | Tab-separated read/write helpers shared by every stage. |

### Why this architecture

- **Blocking-then-scoring** is the standard, tractable way to do entity resolution at this scale,
  and it's literally what the `candidate_pairs.tsv` deliverable expects you to produce.
- **A small embedding model + gradient-boosted trees, not an LLM** — stays inside the
  license/size rule, trains and runs fast enough to iterate many times in a 72-hour window, and is
  easy to explain/debug in the required methodology write-up.
- **Exploiting a verified structural fact** — every Source-2/3 record in the training ground
  truth matches at most one Source-1 entity (checked exhaustively: zero counterexamples) — turns
  Stage 3 from independent per-pair thresholding into an *assignment* problem (each candidate
  record picks its single best Source-1 owner, or none). This directly improves precision, which
  F0.5 weights twice as heavily as recall.
- **Country is used only to narrow blocking, never as a model feature.** France appears only in
  the test set; a model that leaned on country as a feature would be unreliable exactly where it
  can't be locally validated. This is also why blocking, not the classifier, is checked for
  France-specific behavior (see §5).

---

## 3. Methodology in detail

### 3.1 EDA findings that shaped every design decision

- Only **5.6% of Source-1 entities are singletons** (mean ~3.5 matches each) — naively predicting
  "no match" scores badly; recall can't be sacrificed carelessly even under a precision-heavy
  metric.
- **Country agrees in 100% of true matches** in training data — safe to block strictly within
  country.
- **Exact normalized-name equality holds in only ~22%** of true pairs (73% share a first token) —
  blocking needs fuzzy/semantic matching, not just exact-key lookups.
- ~30% of Source-1 business names **recur** (e.g. "Primary Care Group" appears 253 times, each at
  a different address) — name similarity alone can't disambiguate these; address/house-number
  features have to carry that weight.
- Real noise observed: legal-suffix drift (`Pvt`/`Private`, `LLC`/`Limited`, French
  `SAS`/`SARL`/`EURL`/`SCI`), honorifics added/dropped (`Sri`/`Shri`), punctuation junk
  (`***`, `##`, `[..]`), homoglyph typos (`m0tors`, `lbbie`), non-Latin scripts in Source 2/3
  (Devanagari, Gujarati, Kannada) for businesses Source 1 always spells in Latin letters,
  truncated house numbers (`5014` vs `501`), reordered/abbreviated addresses (`Rd`/`Road`, French
  `R.`/`Rue`).

### 3.2 Stage 0 — Normalize

Transliterates non-Latin text to Latin (only rows containing non-ASCII characters pay that cost),
lowercases, strips punctuation noise, expands abbreviations (`St`→`street`, `Rd`→`road`, French
`R.`→`rue`, `Av.`→`avenue`, `Bd.`→`boulevard`, `Ch.`→`chemin`), and produces a legal-suffix- and
honorific-stripped "core" name (`Ram Investment Private Limited` → `ram investment`). House
numbers and PIN/ZIP codes are extracted as a separate signal. Runs at ~6-7 seconds/million rows;
all ~24M records across train+test normalize in a few minutes and are cached so this never reruns
unnecessarily.

### 3.3 Stage 1 — Blocking (candidate generation)

Three techniques, unioned, all restricted to matching country:

1. **Semantic search** with `intfloat/multilingual-e5-small` (118M params, MIT license — well
   under the 8B-parameter cap). Every record's `name | address` becomes a GPU embedding; for each
   Source-2/3 record we retrieve its top-K nearest Source-1 records by cosine similarity (chunked
   matmul + top-k, sized to stay within VRAM regardless of partition size). Catches heavily
   reworded or transliterated names that no lexical key would catch.
2. **Exact-key lexical join** on `(country, name_core)`, as a safety net for obvious matches —
   keys matching more than 40 Source-1 rows are skipped here (left to embeddings + number features
   instead, to avoid candidate-set blowup on generic names).
3. **Exact-key lexical join** on `(country, name_first_token, first_address_number)`, which
   discriminates businesses sharing a generic name but sitting at different addresses.

The union is written directly as `candidate_pairs.tsv` — the exact set fed to Stage 2, with no
further pruning. Blocking quality is measured as **recall@K against the training ground truth**
before ever training the classifier, since this recall is a hard ceiling nothing downstream can
exceed.

### 3.4 Stage 2 — Pairwise matching model

16 features per (Source-1, candidate) pair:

- **Name:** edit-distance ratio and token-set ratio (`rapidfuzz`), on both the raw normalized name
  and the legal-suffix-stripped core name; first-token equality; name length difference.
- **Address:** edit-distance ratio and token-set ratio; house-number/PIN Jaccard overlap, exact
  match, and prefix match (catches truncation like `5014` vs `501`); address length difference.
- **Semantic/context:** the Stage-1 embedding cosine similarity, this candidate's retrieval rank
  among the query's matches, and its margin to the query's best-scoring candidate.
- **Other:** whether the Source-1 name recurs frequently ("generic name, trust the address more"
  signal); whether either side's original text was non-Latin script (transliteration-noise
  signal).

**Country is never a model feature** — only used earlier, to narrow the search.

**Model:** LightGBM binary classifier (gradient-boosted trees; MIT-licensed; not a neural net;
trivially inside the 8B-parameter rule). Trained on a **90/10 split by Source-1 entity** (never by
pair — splitting by pair would leak a Source-1 entity's other candidates across train/validation).
21.9M training rows (all positives + a subsampled 15% of negatives, for training speed); 11.9M
full, unsampled validation rows (needed for a realistic decode-time evaluation).

### 3.5 Stage 3 — Decoding

Each Source-2/3 record is assigned to its single **highest-probability** Source-1 match, kept only
if that probability clears a tuned threshold **τ**. τ is swept directly against the real
competition metric — **macro F0.5 on the held-out validation split**, not AUC/AP, since those are
pairwise metrics that don't capture F0.5's per-entity, precision-weighted nature. Source-1 entities
with no accepted match are correctly left as empty rows (singletons), which score full credit if
genuinely unmatched.

### 3.6 How correctness was checked before ever submitting

10% of training Source-1 entities were held out as a private validation set (never trained on or
used for threshold tuning), scored with the exact macro-F0.5 formula the leaderboard uses.
`utils/validate_submission.py` (the organizers' own validator) was run on every candidate output
before upload, since submissions are rate-limited to 5/day and a formatting rejection wastes one
for nothing.

---

## 4. Problems encountered and how they were handled

- **Blocking recall is the real ceiling, not the model.** Overall blocking recall landed at
  **97.4%** (US 98.6%, **India 95.5%**) — meaning up to ~2.6% of true matches are structurally
  unreachable by Stage 2 no matter how good the classifier is, because they never appear in
  `candidate_pairs.tsv`. India's larger gap is believed to come from heavier transliteration noise
  (romanization doesn't fully normalize honorifics/legal-suffix variants coming from Devanagari
  script) and more free-text address variation (landmark references, missing PIN/state). This is
  the top priority for anyone extending this pipeline — see §7.
- **France is test-only and unseen at training time.** The model can't be validated on France
  locally at all, since no France data exists in `train_*.tsv`. This is why country was excluded
  from every model feature (only used for blocking) and normalization was kept language-agnostic —
  the test-set French singleton rate (4.2%) and match-count distribution ended up closely tracking
  train's US/India numbers, suggesting the model generalized rather than degenerating, but this
  could only be confirmed after running full test-set inference, not validated ahead of time.
- **Generic business names recur constantly** (~30% of Source-1 names, e.g. "Primary Care Group"
  253 times) and are believed to be the dominant source of *false positives* (wrong merges) —
  cases where two different locations of a franchise/chain share a name and the address signal is
  itself weak (missing PIN, landmark-only address). Name similarity features alone cannot resolve
  these; the pipeline leans on house-number/PIN features and the `name-recurs-frequently` feature
  to compensate.
- **Scale forced out-of-core processing everywhere.** 2.2M×10.3M pairs is intractable to hold in
  memory at once even after blocking (Stage 2 alone processes ~120M candidate pairs per split).
  Every stage after blocking is written to run as its **own process**, reading/writing intermediate
  Parquet caches under `work/`, specifically so memory is released between stages rather than
  accumulating in one long-running Python process — see the "Reproducing" instructions in §6 for
  why steps must be run as separate commands, not chained in a notebook.
- **The precision/recall tradeoff had to be tuned against the actual metric, not a proxy.**
  Threshold τ was swept and macro F0.5 climbs from 0.797 at τ=0.30 to a flat peak around
  τ=0.90–0.92, then falls off past τ≈0.93. AUC/AP alone (0.9989 / 0.9868) would not have revealed
  where this peak/plateau actually sits, since those are pairwise, not per-entity, precision-2×
  weighted metrics. τ=0.91 was deliberately picked from the *middle* of the flat region rather than
  the single best point, as a hedge against the test set's true singleton rate differing slightly
  from train's.
- **GPU memory sizing for the embedding similarity search.** A naive full similarity matrix
  between a Source-2/3 chunk and all candidate Source-1 embeddings can exceed VRAM for large
  partitions; `retrieve.py` does the cosine-similarity/top-k search in chunks sized to fit
  available VRAM rather than assuming a fixed batch size works for every partition.

---

## 5. Validated results

Held out 10% of train Source-1 entities, never used in training or threshold tuning:

| Metric | Overall | US | India |
|---|---|---|---|
| Blocking recall (Stage 1 ceiling) | 97.4% | 98.6% | 95.5% |
| Matcher validation AUC | 0.9989 | — | — |
| Matcher validation AP | 0.9868 | — | — |
| **Macro F0.5 (competition metric) at τ=0.91** | **0.9359** | 0.9483 | 0.9167 |

Candidate volume: 119,684,711 candidate pairs for 2.2M train Source-1 entities (~54/entity avg —
a >99.999% reduction from the raw ~10.3M Source-2+3 pool).

**Full test-set inference** (1,732,544 Source-1 entities, including the unseen France partition)
was run end-to-end: 120,781,918 candidate pairs (~70/entity), decoded at the same τ=0.91,
producing 5,531,791 total matched Source-2/3 IDs (all unique, guaranteed by the exclusive-decoder
construction). Test singleton rate is 5.89% overall (US 6.1%, India 6.2%, **France 4.2%**), mean
3.19 matches/entity — closely mirrors train's 5.6% singleton rate / 3.46 mean, and the match-count
distribution shape (peaking at k=3, smooth decay to a max of 11) closely matches the training
ground truth's shape. Both `output/matching_results.tsv` and `output/candidate_pairs.tsv` pass
`utils/validate_submission.py` in both default and `--check-ids` modes.

---

## 6. How to run this on your own system

### 6.1 Requirements

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

Key dependencies: `polars`, `pyarrow`, `numpy` (data processing); `torch` + `sentence-transformers`
+ `transformers` (Stage 1 embeddings — GPU strongly recommended, see below); `rapidfuzz` +
`lightgbm` + `scikit-learn` (Stage 2/3).

`torch` needs a CUDA-enabled wheel to actually use the GPU, e.g.:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

A CPU-only `torch` install also works — `embed.py` falls back to CPU automatically — but is much
slower at this dataset's scale (~24M records total across train+test).

### 6.2 Get the data in place

Place the challenge's provided files so that, relative to the repo root:

```
AmazonML/student_resource/dataset/train/train_source1.tsv
AmazonML/student_resource/dataset/train/train_source2.tsv
AmazonML/student_resource/dataset/train/train_source3.tsv
AmazonML/student_resource/dataset/train/train_ground_truth.tsv
AmazonML/student_resource/dataset/test/test_source1.tsv
AmazonML/student_resource/dataset/test/test_source2.tsv
AmazonML/student_resource/dataset/test/test_source3.tsv
```

`code/business_entity_resolution/src/config.py` derives every other path from its own location —
there is nothing else to configure. This `dataset/` folder is gitignored (host-provided, not ours
to version); get it from the challenge portal.

### 6.3 Reproduce the submission from scratch

Run each stage as its **own process** (not chained in one script/notebook) — this matters on
machines with limited RAM/VRAM, since each stage's memory is released when its process exits,
before the next stage starts. All commands below run from
`code/business_entity_resolution/src/`:

```bash
# 1. Blocking (normalizes + embeds + generates candidates) for train, then test
python run_retrieve.py --split train
python run_retrieve.py --split test

# 2. Pairwise features for train, then test
python run_features.py --split train --n-workers 12
python run_features.py --split test  --n-workers 12

# 3. Split train into a training sample + a full held-out validation set
python split_and_sample.py

# 4. Train the LightGBM matcher, evaluate on validation (AUC/AP)
python train_matcher.py

# 5. Sweep the decoding threshold against the real competition metric
#    (macro F0.5, not AUC) and report per-country breakdown
python evaluate.py

# 6. Score the test candidates with the trained matcher, decode, and write
#    the two submission files into ../../../output/
python run_test_finish.py
```

Each script prints its own timing and is safe to re-run — normalization, embeddings, and
candidate generation are all cached under `work/` (gitignored) and skipped if already present.

**Expected wall time:** roughly 45-60 minutes total end to end for both train and test splits
combined, on the hardware this was developed on (8GB-VRAM GPU, 20 CPU threads, 24GB RAM) —
dominated by GPU embedding of ~24M records and the two ~120M-row feature-computation passes. A
CPU-only setup will take substantially longer at Stage 1.

### 6.4 Validate your output before submitting

```bash
cd AmazonML/student_resource
python3 utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Prints `PASS` (exit 0) when safe to submit, or a numbered list of issues to fix (exit 1). It only
checks format — it does not compute your score. Run it before every leaderboard upload; there are
only 5 submissions/day.

---

## 7. Where to take this next

If you're picking this project up, the highest-leverage next steps, in priority order, are:

1. **Close the India blocking-recall gap (95.5% vs US's 98.6%).** This is the single largest
   known ceiling on the score. Likely worth: fine-tuning or swapping the retrieval embedding model
   on hard negatives from Indian records, or adding a transliteration-normalization pass
   specifically tuned to Hindi/Gujarati/Kannada honorific and legal-suffix variants that the
   current in-code romanization only partially handles.
2. **Investigate false positives on recurring/generic business names** — add features or a
   secondary tie-breaking rule for cases where multiple Source-1 candidates share the same
   `name_core` and the address signal is weak (missing PIN, landmark-only).
3. **Get a real leaderboard score**, not just the offline validation split, and compare per-country
   behavior (especially France, which can only be judged this way, having no training data at all).
4. **Revisit the retrieval `top_k`** used in Stage 1 blocking against the recall/candidate-volume
   tradeoff — more candidates per entity costs Stage 2 compute but raises the recall ceiling.

---

## 8. Repository layout

```
.
├── README.md                               # you are here
├── METHODOLOGY.md                          # original narrative write-up (development-time notes)
├── PS.md, Amazon_PS.pdf, AmazonMLGuidelines.pdf   # original challenge materials
├── AmazonML/student_resource/              # host-provided data (gitignored) + validator + doc template
│   ├── dataset/                            # train/ and test/ TSVs — gitignored, get from challenge portal
│   ├── utils/validate_submission.py        # official output-format validator
│   └── Documentation_template.md           # filled-in methodology submission document
├── code/business_entity_resolution/        # the runnable pipeline
│   ├── src/                                # all stages (see §2 table)
│   ├── README.md                           # pipeline-specific setup/run instructions
│   └── requirements.txt                    # pinned dependencies
├── output/                                 # matching_results.tsv, candidate_pairs.tsv (generated, gitignored)
└── work/                                   # intermediate parquet caches (gitignored, regenerated by the pipeline)
```

`AmazonML/student_resource/dataset/`, `output/`, and `work/` are gitignored — they're either
host-provided (re-downloadable from the challenge portal) or fully regenerable by running the
pipeline in §6.3.
