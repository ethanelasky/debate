# Which CodeContests dataset did the paper's inference runs use?

**Question.** The `nonagentic_logic_and_code` paper's asset section cites **CodeContests+**
(ByteDance-Seed, CC-BY-4.0). The code in `ai-debate` loads **CodeContests-O** (caijanfeng,
Apache-2.0). Only one can be what actually ran.

**Method.** For each problem, take the test cases that were actually used, then fetch that same
problem's test cases from **both** datasets as published on HuggingFace and check which one the
used cases came from. No inference from timings, file sizes, or docstrings — a direct
`(input, output)` set comparison.

**Result.** **71 of 71 used test pairs are present in CodeContests-O. 6 of 71 are in
CodeContests+.** Every problem is 100% covered by CCO. The 6 CC+ hits are trivial collisions on
single-digit inputs.

## Current paired RLVR evaluation artifact

The current `debate` repository does not use the old CCO-only preprocessed file
as its RLVR training set. Training problems and rewards come from the original
`deepmind/code_contests` release at pinned revision
`802411c3010cb00d1b05bad57ca77365a3c699d6`. The reward suite is a seeded sample
of at most 10 `public_tests+private_tests` cases under 500 KB/case and 2 MB total.

Held-out evaluation is a build-time intersection of two independently produced
suites on the same contest-disjoint problems:

| JSONL fields | Source | Role |
|---|---|---|
| `gdm_inputs`, `gdm_outputs` | pinned DeepMind CodeContests | in-distribution |
| `cco_inputs`, `cco_outputs` | pinned CodeContests-O revision `1a765191567b429f633bbd1c6e67b5890dfaf267` | out-of-distribution |

Both suites retain the same deterministic limits: at most 10 cases, 500 KB per
case, and 2 MB total per problem. `scripts/fetch_cco_eval.py` creates the CCO
staging extraction; `scripts/build_codecontests_paired_eval.py` joins it with
the GDM held-out rows and writes the self-contained runtime artifact:

```bash
python scripts/fetch_cco_eval.py
python scripts/build_codecontests_paired_eval.py
```

`paired_test.manifest.json` records both upstream revisions and input hashes,
the suite policies and caps, the output hash, and rating-slice counts. The
current artifact has 456 paired problems; the inclusive Codeforces `[800,1000]`
slice has exactly 55 and `[800,1100]` has 75. Production configs assert these
counts at load time.

At evaluation, one temperature-zero completion is extracted once and executed
against both suites. W&B receives `eval/gdm_correct`,
`eval/gdm_tests_passed_frac`, `eval/cco_correct`, and
`eval/cco_tests_passed_frac`; `eval/correct` remains a backward-compatible GDM
alias. CCO never enters the RLVR training reward. ByteDance CodeContests+ is not
used by this pipeline.

---

## The datasets

| | HuggingFace | License | Test field |
|---|---|---|---|
| **CodeContests-O** (CCO) | [`caijanfeng/CodeContests-O`](https://huggingface.co/datasets/caijanfeng/CodeContests-O) | Apache-2.0 | `corner_cases` |
| **CodeContests+** (CC+) | [`ByteDance-Seed/Code-Contests-Plus`](https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus) | CC-BY-4.0 | `test_cases` |

The paper's bib entry `wang2025codecontestsplus` → arXiv 2506.05817 → CC+. CCO (arXiv 2601.13682)
is not cited anywhere in the paper.

---

## The certificate

"Used" = the `(input, output)` pairs in `ai_debate/data/codecontests/train.jsonl`, the preprocessed
file the loader reads. CCO column = `corner_cases` fetched from `caijanfeng/CodeContests-O`
train shard 0. CC+ column = `test_cases` fetched from `ByteDance-Seed/Code-Contests-Plus`,
config `ccplus_1x`.

| problem | used | CCO has | CC+ has | used pairs found in CCO | used pairs found in CC+ |
|---|---:|---:|---:|---:|---:|
| `584_B. Kolya and Tanya` | 10 | 28 | 17 | **10/10** | 3/10 |
| `975_C. Valhalla Siege` | 5 | 42 | 27 | **5/5** | 0/5 |
| `55_D. Beautiful numbers` | 10 | 16 | 28 | **10/10** | 1/10 |
| `13_E. Holes` | 2 | 37 | 24 | **2/2** | 1/2 |
| `818_A. Diplomas and Certificates` | 10 | 13 | 12 | **10/10** | 1/10 |
| `772_A. Voltage Keepsake` | 10 | 43 | 29 | **10/10** | 0/10 |
| `1240_B. Sequence Sorting` | 5 | 34 | 32 | **5/5** | 0/5 |
| `1509_B. TMT Document` | 9 | 33 | 18 | **9/9** | 0/9 |
| `p01171 Everlasting...?` | 10 | 32 | 27 | **10/10** | 0/10 |
| **TOTAL** | **71** | | | **71** | **6** |

Every used pair is in CCO. No problem exceeds 30% coverage by CC+, and most are 0%.

### Worked example — `584_B. Kolya and Tanya`

The two datasets carry different suites for the same problem:

```
CCO first inputs:  ['2', '3', '99999', '99991', '99990', '10000']
CC+ first inputs:  ['1', '2', '3', '4', '5', '6']

CCO distinct inputs: 28      CC+ distinct inputs: 17      intersection: 6
```

The 10 used inputs are a **subset of CCO's** and **not a subset of CC+'s**.

### A second, independent check — dataset ordering

`preprocess_codecontests.py` writes problems "in dataset order (deterministic)". CCO train shard 0
holds 9 problems. Their names match rows 0–8 of `train.jsonl` **exactly, in order**:

```
i   CodeContests-O train shard 0        data/codecontests/train.jsonl row i
0   584_B. Kolya and Tanya              584_B. Kolya and Tanya
1   975_C. Valhalla Siege               975_C. Valhalla Siege
2   55_D. Beautiful numbers             55_D. Beautiful numbers
3   13_E. Holes                         13_E. Holes
4   818_A. Diplomas and Certificates    818_A. Diplomas and Certificates
5   772_A. Voltage Keepsake             772_A. Voltage Keepsake
6   1240_B. Sequence Sorting            1240_B. Sequence Sorting
7   1509_B. TMT Document                1509_B. TMT Document
8   p01171 Everlasting...?              p01171 Everlasting...?
```

9/9. The preprocessed file is CCO in CCO's own row order.

### Reproducing it

```python
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq, json
fs = HfFileSystem()

# CodeContests-O
with fs.open('datasets/caijanfeng/CodeContests-O/data/train-00000-of-01386.parquet','rb') as fh:
    d = pq.ParquetFile(fh).read(columns=['name','corner_cases']).to_pydict()
cco = {n.strip(): {(c['input']['stdin'].strip(), c['output']['stdout'].strip())
                   for c in cc} for n, cc in zip(d['name'], d['corner_cases'])}

# CodeContests+  (584_B lives in ccplus_1x/part-00015-of-00020.parquet, row group 2)
with fs.open('datasets/ByteDance-Seed/Code-Contests-Plus/ccplus_1x/part-00015-of-00020.parquet','rb') as fh:
    t = pq.ParquetFile(fh).read_row_group(2, columns=['id','test_cases'])
i = t.column('id').to_pylist().index('584_B')
ccp = {(c['input'].strip(), c['output'].strip()) for c in t.column('test_cases')[i].as_py()}

row = next(json.loads(l) for l in open('ai_debate/data/codecontests/train.jsonl')
           if l.strip() and json.loads(l)['name'].startswith('584_B'))
used = {(a.strip(), b.strip()) for a, b in zip(row['inputs'], row['outputs'])}
print(len(used & cco['584_B. Kolya and Tanya']), '/', len(used))   # 10 / 10
print(len(used & ccp), '/', len(used))                             # 3 / 10
```

---

## Where the dataset choice enters the code

The HF identifier appears in **exactly one place**. No YAML or config names it.

`papers/nonagentic_logic_and_code/scripts/neurips/preprocess_codecontests.py:90`

```python
ds = load_dataset("caijanfeng/CodeContests-O", split=split)
...
corner_cases = row.get("corner_cases", [])     # CCO's field; CC+ calls its field `test_cases`
```

The runtime loader never touches HuggingFace — it reads the preprocessed file:

`data_loader/codecontests_loader.py:136`

```python
DEFAULT_JSONL_PATH = "data/codecontests/test.jsonl"
```

Two further CCO-specific markers in that loader:

- module docstring: *"contains problems from `caijanfeng/CodeContests-O`"*
- line 299: `"cf_rating_min/cf_rating_max are ignored: CodeContests-O has no CF ratings."`

**No file anywhere in `~/code` references CC+ outside the paper's `.tex` sources.** The HF cache on
the machine has `datasets--caijanfeng--CodeContests-O` and `datasets--deepmind--code_contests`;
there is no ByteDance-Seed entry.

---

## The test-size reduction is real, and it is in the CCO path

`preprocess_codecontests.py`:

```python
MAX_TOTAL_IO_BYTES    = 10_000_000   # 10 MB total test I/O per problem
MAX_SINGLE_TEST_BYTES =    500_000   # 500 KB for any single case
MIN_TEST_CASES        = 1
MULTI_ANSWER_PHRASES  = [...]        # exact comparison can't grade these
```

> Total I/O size capped (problems with massive test inputs are typically competitive programming
> tasks requiring C++ for performance)

It was necessary — **CCO's train split is 692 GB** (324 GB download). Problems are dropped whole
when they exceed a cap; surviving problems keep all their cases.

There is a second, unrelated size fix on the RL side: `scripts/build_codecontests_frozen_public.py`
reads only projected columns from DeepMind's parquet, producing a 26 MB artifact of 9,317 problems.
Two fixes, two pipelines, neither involving CC+.

---

## The debate runs behind the paper's CodeContests numbers

From `tools/critic_review/_classifications/`:

```
2026-04-21_11-08-23.892792_CodeContests_Standard_Single_Proposer_Debate_bo1_qwen35_122b_35b_1k
2026-04-21_11-08-23.892792_CodeContests_Standard_Single_Proposer_Debate_bo1_qwen35_122b_35b_1k__pc_nt
2026-04-21_14-10-01.194431_CodeContests_Standard_Single_Consultancy_bo1_qwen35_122b_35b_1k
2026-04-21_14-30-40.799254_CodeContests_Standard_Single_Proposer_Debate_bo1_gpt_oss_120b_20b_1k
2026-04-21_14-30-40.799254_CodeContests_Standard_Single_Proposer_Debate_bo1_gpt_oss_120b_20b_1k__pc_nt
2026-04-21_15-19-36.431990_CodeContests_Single_No_Transcript_bo1_qwen35_122b_35b_1k
2026-04-21_17-30-37.100912_CodeContests_Standard_Single_Consultancy_bo1_gpt_oss_120b_20b_1k
2026-04-21_19-44-17.603470_CodeContests_Single_No_Transcript_bo1_gpt_oss_120b_20b_1k
2026-04-21_22-51-02.726737_CodeContests_Standard_Single_Proposer_Debate_bo1_qwen35_35b_qwen3_4b_1k
2026-04-21_22-51-02.726737_CodeContests_Standard_Single_Proposer_Debate_bo1_qwen35_35b_qwen3_4b_1k__pc_nt
2026-04-22_00-21-08.681336_CodeContests_Standard_Single_Consultancy_bo1_qwen35_35b_qwen3_4b_1k
2026-04-22_04-46-48.321102_CodeContests_Single_No_Transcript_bo1_qwen35_35b_qwen3_4b_1k
```

Proposer accuracy, from `analysis/2026_neurips_per_question_results.csv` (~988 problems per cell):

```
Qwen3.5-122B  73.7%      gpt-oss-120B  67.5%      Qwen3.5-35B  65.2%
```

Linking the runs to the preprocessed file: the answer cache
(`ai_debate/.cache/answers/CODE_CONTESTS/`, 12,436 records) stores each graded program and its
verdict. Across 262 records where a program failed and the record exposes the expected output,
that expected output is found in the preprocessed suite **228 times (87%)**; the misses are a
superset effect — the runs graded 10 cases per problem where the surviving extract kept 8–10.

---

## Scope and limits

- The certificate proves the **preprocessed dataset file in the repo is CodeContests-O**. The exact
  ~1k-problem eval file the runs read (`train_full.jsonl`, referenced in the preprocessor's usage
  examples) is no longer on disk, so it cannot be compared directly. It was produced by the same
  preprocessor, which has exactly one HF source.
- The 9 certificate problems have no failing run records (models solved them), so the
  cache→dataset link rests on the 87% corpus-wide match above rather than on those 9.
- All evidence is from one machine. It establishes what ran there.

**Two earlier arguments are withdrawn.** Verifier timing (median 47 ms) and test-case counts do not
discriminate: a CC+ run through the same preprocessor, with the same 500 KB / 10 MB caps and
unpaired-case dropping, would also produce small, fast, ~10-case suites. Only direct
`(input, output)` comparison against both published datasets settles it.

---

## Consequence for the paper

If this holds, the asset section misattributes both dataset and license:

- cites **CodeContests+** — CC-BY-4.0, ByteDance-Seed release
- what ran was **CodeContests-O** — Apache-2.0, `caijanfeng`, arXiv 2601.13682

Worth correcting before camera-ready, along with the bib entry.

## What would overturn this

1. A run directory for a `..._CodeContests_..._1k_...` run whose config records a CC+ path.
2. The intermediate `train_full.jsonl` — if its rows carry `test_cases` / `true_positive_rate` it
   is CC+; if `corner_cases` / `num_corner_cases`, CCO. Or run the table above against it.
3. `ls ~/.cache/huggingface/hub | grep -i contests` on another machine showing a ByteDance entry.
