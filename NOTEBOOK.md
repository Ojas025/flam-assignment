# Experiments

---

## Observation
- In `fertility.py`, the random seed doesn't actually help in reproducibility as there are no random functions involved.

## E1 - **split() vs split(" ")**
**Hypotheses:** Double spaces create distinct empty words, understating the fertility on affected lines. Fertility shows a dip if double spaces exist within the corpus, since fertility is tokens/words - the affected lines report fewer tokens/words.

**Prediction:** Switching to `split()` instead, collapses the redundant whitespaces - dropping the extra words and thus increasing fertility for the affected lines

### Run 1
**Experiment:** Ran the word split ablation on the sample corpus, with gpt2 tokenizer, and everything else held constant, so any delta is caused by the split alone.

`python partA/run_experiments.py --experiment split_space --dataset sample --no-plot`

**Result:** 
- English: 1.2652 → 1.2831 (+1.41% delta, 79 → 78 words, 1 empty word, 10% of lines affected)
- Hindi: 7.4485 → 7.5985 (+2.01% delta, 62 → 61 words, 1 empty word, 10% of lines affected)

**Verdict:** Confirmed live bug in v0. The artifact `fertility.py` only applied `strip()` and NFC normalization, never collapsing whitespace. The reported v0 figures (eng 1.27, hin 7.45) directly incorporated this bug.

### Run 2
**Experiment:** Ran the same ablation on raw FLORES-200 dev - 997 sentences in each of 6 languages - to see whether the effect survives outside the seeded sample, and how it affects cross-lingual comparisons.

`python partA/run_experiments.py --experiment split_space --dataset flores --no-plot --raw`

**Result:** 
- English: +0.00% delta (1 empty word, 0.1% lines affected)
- Hindi: +0.04% delta (9 empty words, 0.7% lines affected)
- Tamil: +0.54% delta (101 empty words, 8.3% lines affected)
- Malayalam: +0.70% delta (132 empty words, 7.5% lines affected)
- Telugu: +1.10% delta (210 empty words, 11.2% lines affected)
- Kannada: +3.60% delta (692 empty words, 30.0% lines affected)
- Compression (`tok/char`) delta: 0.00% across all languages (character length is invariant to word splitting).

**Verdict:** The bug introduces cross-language bias. Because the rate of double spaces varies widely across languages in raw text (virtually absent in	English at 0.1%, but present in 30.0% of Kannada sentences), `split(" ")` decreases fertility in a language-dependent manner.
 
### Run 3
**Experiment:** Ran the split ablation on FLORES-200 and the sample corpus after passing through our new preprocessing pipeline (`re.sub(r"\s+", " ", ...)`)

`python partA/run_experiments.py --experiment split_space --dataset flores --no-plot`
`python partA/run_experiments.py --experiment split_space --no-plot`

**Result:**  
- FLORES-200: 0 empty words, 0 affected lines, Δ 0.00% across all 6 languages.
- Sample Corpus: Affected lines dropped 10% → 0%, and the +1.41% / +2.01% gap collapsed to 0.00%.

**Verdict:** The corrected preprocessing pipeline successfully neutralizes the vulnerability before measurement. However, this is a property of the new pipeline, not of v0. The v0 artifact lacked whitespace collapsing, so the flaw was active in all reported baseline deck figures.

**Observation:**
- When run on `MASSIVE` corpus, which already collapses extra whitespaces, the effect of this experiment showed 0% delta.
- Hence, this essentially becomes a property of the preprocessing pipeline - thus making it not a critical bug, but still being reported wrong in the `REPORT_v0`

---

## E2 - A2 - **line.lower() vs preserve casing**

**Hypotheses:** `line.lower()` does not measure realistic tokenizer behavior for case-sensitive language such as English

**Prediction:** Should only affect scripts where casing matters, asymmetric measure across languages. GPT2 tokenizer being case-sensitive, lowercasing leads to fragmentation for common capitalized words - thus increasing token count, leading to inflated fertility & compression while leaving Hindi untouched.

**Experiment:** Ran the lowercasing ablation on the sample corpus - the artifact's line.lower() vs case-preserved text, gpt2 tokenizer, everything else held constant, so any delta is caused by the casing alone.

`python partA/run_experiments.py --experiment lowercase --dataset sample --no-plot`

**Result** 
- English: 100% of lines modified by `lower()`, 30% of lines (3 of 10) tokenize into more tokens. Total tokens: 96 → 99 (+3 tokens). Measured fertility: 1.2293 → 1.2652 (+2.84%), tok/char: 0.2196 → 0.2256 (+2.7%).
- Hindi: 0% text changed, tokens remain 459 → 459, all deltas 0.00%.

**Verdict** Prediction holds - asymmetric as expected, affects only English text. Preserving casing reflects real text distribution. So the report overstates English fertility (1.27 vs 1.23) and understates the hin/eng ratio (5.89 vs 6.06). This result is based on the tokenizer being case-sensitive.

- **Observed experiment:** Decoded gpt2's token strings on the sample's English lines,
raw vs lowercased, to see what the extra tokens actually are.
Whole cased words exist as single tokens and split when lowercased:
`Thursday` → `th`+`ursday`, `NASA` → `n`+`asa`, ` GPU` → ` g`+`pu`.

---

## E3 - A2 - **macro vs micro aggregation**

**Hypotheses:** Macro aggregation measures average sentence-level density by assigning equal weight to every segment regardless of length, whereas micro aggregation computes aggregate corpus-level density by pooling tokens over total units. When segment-level token densities vary across segments, macro and micro can diverge because they assign different weights to each segment; micro is the mathematically appropriate choice for estimating overall corpus-level token density.

**Prediction:**
1. Micro-averaging will diverge from macro-averaging, the magnitude and direction of the difference will depend on the distribution of segment length and token-density.
2. For tokens per segment (`tps`), where each line has a denominator identically equal to 1, macro and micro will coincide exactly.
3. Even with aligned parallel sentences, macro and micro may produce different cross-language comparisons because translations can differ in length and tokenization density.

### Run 1: Starter-Kit Sample Corpus (10 lines, `gpt2`)
**Experiment:** Evaluated macro vs. micro aggregation across `tpc` (tokens/codepoint), `tps` (tokens/segment), `tpb` (tokens/byte), and `fert` (tokens/word) on the 10-line sample corpus.

`python partA/run_experiments.py --experiment macro_micro --dataset sample --no-plot`

**Result:**
- **English:**
  - `tpc`: Macro 0.2256 vs Micro 0.2210 (-2.06% delta)
  - `tpb`: Macro 0.2256 vs Micro 0.2210 (-2.06% delta)
  - `tps`: Macro 9.90 vs Micro 9.90 (+0.00% delta)
  - `fert`: Macro 1.2652 vs Micro 1.2532 (-0.95% delta)
- **Hindi:**
  - `tpc`: Macro 1.5791 vs Micro 1.5828 (+0.23% delta)
  - `tpb`: Macro 0.5990 vs Micro 0.6008 (+0.30% delta)
  - `tps`: Macro 45.90 vs Micro 45.90 (+0.00% delta)
  - `fert`: Macro 7.4485 vs Micro 7.4032 (-0.61% delta)
- **Cross-Language Ratios (Hindi / English):**
  - Fertility Ratio: Macro 5.8871 vs Micro 5.9076 (+0.35% shift)
  - Tok/Char (`tpc`) Ratio: Macro 6.9991 vs Micro 7.1620 (+2.33% shift)

### Run 2: FLORES-200 dev (997 parallel sentences, `gpt2`)
**Experiment:** Evaluated macro vs. micro aggregation across 997 parallel sentences per language for English, Hindi, Kannada, Malayalam, Tamil, and Telugu.

`python partA/run_experiments.py --experiment macro_micro --dataset flores --no-plot`

**Result:**
- **Fertility:** English -0.67% (1.2825 vs 1.2740); Hindi -0.38% (7.8256 vs 7.7959); Kannada -1.21% (22.9033 vs 22.6254); Malayalam -1.24% (27.4295 vs 27.0892); Tamil -1.00% (24.8610 vs 24.6119); Telugu -0.67% (20.6123 vs 20.4739).
- **Tok/Char (`tpc`):** English -0.89% (0.2152 vs 0.2132); Hindi +0.07% (1.5277 vs 1.5287); Kannada 0.00% (2.6638 vs 2.6638); Malayalam +0.05% (2.7423 vs 2.7435); Tamil +0.04% (2.7182 vs 2.7193); Telugu -0.04% (2.6439 vs 2.6427).
- **Tokens/Segment (`tps`):** Exactly +0.00% across all 6 languages.
- **Ratio Shift:** FLORES Hin/Eng fertility ratio shifts from 6.1016 (macro) to 6.1193 (micro) (+0.29%).

### Verdict
- Macro and micro measure different quantities. Macro represents average sentence-level token density, while micro represents aggregate corpus-level token density. Neither is inherently incorrect.
- Differences were measurable but relatively small. Maximum individual-metric divergence was 2.06% on the sample corpus and 1.24% on FLORES.
- Macro and micro TPS were identical across all languages and datasets, as expected.
- Aggregation choice changed the Hindi/English fertility ratio by +0.35% on the sample and +0.29% on FLORES. The TPC ratio shifted by +2.33% on the sample.
- Micro fertility was lower than macro fertility across all six FLORES languages, while TPC showed small differences in both directions. There is no universal direction of divergence.
- Sentence-level comparisons should use macro when every aligned sentence deserves equal weight. Corpus-level token consumption comparisons should use micro.

---

## E4 - A2 - **compression formula correction**

**Hypotheses:** Character counts don't necessarily represent equivalent amounts of semantic information, so Hindi having higher `TPC` doesn't tells us how much expensive it is to represent the same meaning. Since in parallel text, each segment represents approximately the same semantic content. Instead of comparing the character counts, we compare the number of tokens required to encode each equivalent segment.
TPC is valid for measuring token density relative to unicode codepoints, to compare tokenizers on the same language.
It cannot independently establish cross-language semantic token cost because codepoint counts are not equivalent across languages.

**Prediction:** 
- Because Indic scripts require multi-codepoint clusters and 3-byte UTF-8 representations per character, codepoint- and grapheme-based ratios will show greater apparent penalties than byte-level or segment-level metrics.
- Tokens per aligned segment will provide a more direct estimate of relative token consumption for approximately equivalent semantic content, for a given tokenizer.

### Run 1

**Experiment:** Evaluated micro-aggregated token counts across all 5 candidate denominators on the 10-line sample corpus under gpt2, alongside comparison with Indic tokenizers.

`python partA/run_experiments.py --experiment compression --dataset sample --no-plot`

**Result**
Ratios to English:
- Hindi: tok/byte 2.72x | tok/cp 7.16x | tok/graph 10.65x | tok/word 5.93x | tok/seg 4.64x

### Run 2

**Experiment:** Evaluated micro-aggregated token counts across all 5 candidate denominators on the 10-line sample corpus under indic tokenizers

`python partA/run_experiments.py --experiment compression --dataset sample --no-plot --tokenizers all`

**Result:**
- hf:xlm-roberta-base: Ratios (Hin/Eng): tok/byte 0.52x | tok/cp 1.36x | tok/graph 2.02x | tok/word 1.13x | tok/seg 0.88x
- hf:sarvamai/sarvam-1: Ratios (Hin/Eng): tok/byte 0.46x | tok/cp 1.21x | tok/graph 1.80x | tok/word 1.00x | tok/seg 0.79x 

**Verdict**
- The results demonstrate that cross-lingual tokenization overhead is strongly tokenizer-dependent, rather than a fixed property of a language or script.
- With GPT-2, Hindi requires 4.64× as many tokens per aligned segment as English. With XLM-R, this falls to 0.88×, and with Sarvam-1, to 0.79×.
- The approximately 7× Hindi/English TPC ratio observed with GPT-2 does not translate into a corresponding semantic-content-normalized token cost. Codepoint-based ratios measure Unicode-level token density, not the cost of representing equivalent meaning.
- Recommended conventions:
	- Cross-Lingual Serving Cost: Measure as tokens per parallel segment (tok/seg) on parallel text.
 	- Encoding Efficiency: Measure as UTF-8 bytes per token (bytes/tok).
  - Within-Language Comparison: Retain tok/cp strictly for evaluating multiple tokenizers on the same language, never for cross-lingual comparisons.

---

## E5 - A2 - **arg order dependency**

**Hypotheses:** `fertility.py` selects its comparison baseline via `base = langs[0]`, making the reported cross-language ratio and verdict ("worse" vs "better") an artifact of CLI argument order rather than an intrinsic property of the corpora.

**Prediction:** Swapping the order of `--corpus` arguments on identical data will invert the numeric ratio and flip the verdict from "worse" to "better", should still preserve the underlying numbers - only reporting difference.

**Experiment:** Ran the argument order ablation across all ordered pairs on both the starter-kit sample corpus and FLORES-200 dev with `gpt2`.

`python partA/run_experiments.py --experiment arg_order --dataset sample --no-plot`

`python partA/run_experiments.py --experiment arg_order --dataset flores --no-plot`

**Result** 
On sample corpus:
	- hin is 5.89x the fertility of eng (worse tokenization), `ratio: 5.8871`
	- eng is 0.17x the fertility of hin (better tokenization), `ratio: 0.1699`

On FLORES-200:
	- Cross-lingual comparisons confirm algebraic reciprocity across all pairs

**Verdict**
1. This is a reporting flaw, not a data corruption bug. Per-language fertility and compression values are completely invariant to argument order.
2. Inverted Framing:** "Hindi is 5.89× English (worse)" and "English is 0.17× Hindi (better)" express the exact same mathematical relationship
3. For > 2 languages, defaulting to `langs[0]` outputs an incomplete comparison set and risks misleading readers if the implicit baseline is not the intended reference.
4. Multi-corpus evaluation should either compute an explicit pairwise matrix or accept an explicit `--baseline <lang>` flag
5. Resolution:
	- An explicit `--baseline <lang>` CLI flag
	- A full pairwise ratio matrix for multi-language evaluation
	- A unit test asserting `ratio(a, b) * ratio(b, a) == 1.0`

---

## E6 - A2 - **tokenizer dependency**

**Hypotheses:** The 5.89× fertility penalty reported in `REPORT_v0.md` is a tokenizer-specific artifact of evaluating Indic text with an English-centric BPE tokenizer (`gpt2`), rather than an intrinsic property of Indic scripts.

**Prediction:** Tokenizers trained on multilingual or Indic-specialized corpora will represent Hindi with	fertility near parity with English (~1.0×), disproving the claim that "any tokenizer will struggle" and collapsing serving cost projections for Indic languages.

### Run 1

**Experiment:** Evaluated the 10-line sample corpus across five tokenizers: `gpt2`, `xlm-roberta-base`, `google/mt5-small`, `google/muril-base-cased`, and `sarvamai/sarvam-1`.

`python partA/run_experiments.py --experiment tok_dependence --tokenizers all --dataset sample --no-plot`

**Result**
- `gpt2`: Eng 1.27 tok/w, Hin 7.45 tok/w (Ratio: **5.89×**) | Hin tokens: 459
- `xlm-roberta-base`: Eng 1.28 tok/w, Hin 1.42 tok/w (Ratio: **1.10×**) | Hin tokens: 88
- `mt5-small`: Eng 1.45 tok/w, Hin 2.06 tok/w (Ratio: **1.42×**) | Hin tokens: 125
- `muril-base-cased`: Eng 1.23 tok/w, Hin 1.23 tok/w (Ratio: **1.00×**) | Hin tokens: 76
- `sarvam-1`: Eng 1.36 tok/w, Hin 1.36 tok/w (Ratio: **1.00×**) | Hin tokens: 84
- Corpus control: English averages 5.74 chars/word vs Hindi 4.75 chars/word (0.83×).

### Run 2

**Experiment:** Scaled the ablation to FLORS-200 --- 997 parallel sentences across English, Hindi, and 4 Dravidian languages (Kannada, Malayalam, Tamil, Telugu).

`python partA/run_experiments.py --experiment tok_dependence --tokenizers all --dataset flores --no-plot`

**Result:**
- **Hindi Fertility Ratio:** Drops from **6.10×** (`gpt2`: 7.83 vs 1.28) to **0.97×** on `muril` (1.25 vs 1.29) and **0.96×** on `sarvam-1` (1.40 vs 1.46). Hindi requires *fewer* tokens/word than English under Indic tokenizers.
- **Dravidian Languages:**
  - Under `gpt2`, fertility explodes to 16.07× (Telugu, 20.6 tok/w), 17.86× (Kannada, 22.9 tok/w), 19.38× (Tamil, 24.9 tok/w), and 21.39×
  (Malayalam, 27.4 tok/w) due to raw byte fallback.
  - Under `muril` and `sarvam-1`, Dravidian fertility collapses by ~90% to 1.7–2.8 tok/w (1.35×–1.94× English).
- **Normalized Compression (`tok/char`):**
  - On `muril`, Tamil is **0.192 tok/char** vs English **0.217 tok/char** (0.88× — Tamil produces 12% *fewer* tokens per character than
  English).
  - Kannada is 0.99×, Malayalam is 1.01×, and Telugu is 1.17× of English tok/char.

**Verdict**
- `REPORT_v0.md` asserted that *"Hindi simply has more Unicode characters per word, so any tokenizer will struggle. This is a property of the script, not the tokenizer."* This is refuted at scale: Hindi has fewer characters/word (5.10 vs 5.97), and  Indic tokenizers achieve exact parity with English (0.96×–1.00×)
- Even though the indic-language cost can be neutralized using an Indic tokenizer, a tokenizer cannot be replaced or modified in isolation as it is coupled with the LLM.
- If leadership decides to pay the fixed infrastructure overhead of deploying an Indic-specialized model, **the serving cost  Hindi on that model is 1.0× (equal to English)**.
- Budgeting 6× variable capacity *on top of* running a dedicated Indic model double-counts the penalty and over-provisions server capacity by **~500%–600%**.

---

## E7 - A2 - **final metrics**
**Hypotheses:** The corrected pipeline (whitespace collapse, preserved casing, `split()`, explicit baseline) will change every number REPORT_v0 printed from the same 10 lines, and the added denominators will disagree with each other — so v0's "two metrics agree → robust" was never a real check.

**Prediction:**
The five denominators give five different hin/eng ratios instead of v0's "agreeing" 5.89× / 7.0×.

**Experiment:** Ran the corrected benchmark on the sample corpus — gpt2 (v0's tokenizer) plus muril, baseline eng, five denominators:

`python partA/updated_fertility.py`

`python partA/updated_fertility.py --baseline hin` (order-independence check)

**Result:**
Same 10 lines, changed numbers vs REPORT_v0 (gpt2, micro-averaged):

| metric | REPORT_v0 | corrected |
|---|---|---|
| eng fertility (tok/word) | 1.27 | 1.2179 |
| hin fertility (tok/word) | 7.45 | 7.5082 |
| hin/eng fertility ratio | 5.89× | 6.16× |
| eng tok/char | 0.226 | 0.2125 |
| hin tok/char | 1.579 | 1.5848 |
| tok/char ratio | 7.0× | 7.46× |

Ratios v0 never reported (hin/eng):

| denominator | gpt2 | muril |
|---|---|---|
| tok/word | 6.16× | 1.07× |
| tok/graph | 11.11× | 1.92× |
| tok/byte | 2.82× | 0.49× |
| tok/seg (TPS) | 4.82× | 0.84× |
| tok/cp | 7.46× | 1.29× |

- Alignment check passed before any TPS was printed: 10 = 10 non-empty segments.
- Baseline is a flag, not argument order: `--baseline hin` flips the table — gpt2 reports eng as 0.16× hin, the reciprocal of 6.16× (E5 fix confirmed).
- Absolute tokens per sentence: gpt2 eng 9.50 vs hin 45.80; muril eng 9.10 vs hin 7.60.

### Run 2 — FLORES-200: tokens per sentence for the Dravidian languages
The sample corpus holds only English and Hindi, so it has no tokens-per-sentence figures for Kannada, Tamil, Telugu, or Malayalam — Dravidian cost can't be budgeted from it. Ran the same tool on FLORES (997 parallel sentences × 6 languages):

`python partA/updated_fertility.py --corpus flores`

Tokens per parallel sentence (TPS):

| tokenizer | eng | hin | kan | tam | tel | mal |
|---|---|---|---|---|---|---|
| gpt2 | 25.82 | 192.40 | 350.13 | 398.26 | 336.44 | 390.62 |
| muril | 26.44 | 30.74 | 28.03 | 27.87 | 32.13 | 30.95 |

- TPS ratio vs English — gpt2: hin 7.45×, kan 13.56×, tam 15.43×, tel 13.03×, mal 15.13×.
- Same six languages under muril: 1.16×, 1.06×, 1.05×, 1.22×, 1.17× — near parity across the board.
- Alignment re-checked on this corpus: 997 = 997 segments in all six languages.

**Verdict:**
- Every cell of v0's sample table changes under the corrected pipeline: eng fertility 1.27 → 1.2179 (E2's predicted ~1.23 confirmed — v0 lowercased English) and tok/char 0.226 → 0.2125; hin moves under 1% (7.45 → 7.5082) because casing doesn't affect Devanagari.
- v0's "two metrics agree (5.89× and 7.0×), so it's robust" collapses: on identical text the five denominators span 2.82× to 11.11×. Those two numbers were two ratios of the same token count, not independent measurements.
- The routing number flips with the tokenizer: per sentence, Hindi costs 4.82× English on gpt2 but 0.84× on muril — cheaper than English. "Budget 6× for Hindi" is a property of gpt2 (E6), not of Hindi.
- TPS is the denominator recommendation: it is the only one that holds content constant across languages, and Run 2 gives a per-sentence cost for every language — Dravidian budgets at 13–15× English on gpt2, or 1.05–1.22× on muril.
