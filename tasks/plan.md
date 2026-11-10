# Implementation Plan: "The Audit" — Flam AI Intern Take-Home

## Overview

Audit the previous intern's `REPORT_v0.md` (tokenizer fertility + serving
capacity claims) and produce a graded submission: measured evidence for every
claimed flaw, a corrected multilingual tokenizer analysis, a capacity
reconciliation, and a constrained decision memo — plus NOTEBOOK.md,
AI_USAGE.md, and preparation for a live defense. Target: ~10 focused hours
over 5 days. Graded on evidence discipline, not volume.

## Research findings (verified — sources at bottom)

### Environment (local, verified)
- Python 3.13.14; **`tiktoken` installed**; **`regex` installed** (2026.9.29 → grapheme clusters via `\X` ready).
- **`transformers`, `datasets`, `sentencepiece` NOT installed** → `pip install transformers sentencepiece` needed for HF tokenizers (XLM-R, mT5 use SentencePiece). `datasets` optional if we read the FLORES tarball directly (recommended: fewer deps).
- HuggingFace API reachable (model downloads will work).

### FLORES-200 (A1 corpus source)
- **Public download, no login:** `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` — only **25.6 MB** (verified HEAD: content-length 25585843). Note: HF `facebook/flores` is **gated** (requires accepting conditions + login) — avoid; use the fbaipublicfiles tarball. Newer official version: `openlanguagedata/flores_plus` (OLDI), but the classic tarball is sufficient and cited by the assignment.
- **997 dev / 1012 devtest** parallel sentences per language; ~21 words/sentence average; domain = **wikinews** (842 web articles, 3001 sentences total across splits); license **CC-BY-SA 4.0**; aligned by `id` across languages (all configs share sentence ids).
- Language codes (all confirmed present in flores200 README): `eng_Latn`, `hin_Deva`, `kan_Knda`, `tam_Taml`, `tel_Telu`, `mal_Mlym`, `ben_Beng`, `mar_Deva` — covers both Part A (4+ langs) and Part C (all 6 product languages).
- Paper: NLLB, "No Language Left Behind", arXiv:2207.04672.

### Tokenizers for A3 (availability verified via HF API, all HTTP 200, ungated)
- `xlm-roberta-base` — 100 languages incl. all Indic (model card verified).
- `google/mt5-small` — 101 languages (model card verified); 250k vocab.
- ~~`ai4bharat/indic-bert`~~ — **GATED (401 on file download)** despite HF API metadata returning 200; discovered during env setup. Use **`google/muril-base-cased`** instead — verified working, Indic-specific (23 Indic languages incl. all Part C languages).
- `gpt2` via tiktoken — the intern's baseline.
- ⚠️ **"Aksharagarbha" does NOT exist on HuggingFace** (API search returns empty) — earlier session's suggestion was a hallucination. Do not cite it.
- `vishnubert/...` is gated (401) — avoid.

### vLLM serving metrics (B4 — from docs.vllm.ai/en/latest/usage/metrics/)
- **`vllm:num_preemptions`** — cumulative preemption counter. Expected: 0 through batch 24; ≥7 at batch 32; ≥23 at batch 48 (log's `preempted_seqs` counts *distinct sequences* preempted ≥ once; this counter counts *events*, so ≥ those values).
- **`vllm:kv_cache_usage_perc`** — expected pinned ≥0.93 at batch ≥24.
- **`vllm:prompt_tokens` and `vllm:generation_tokens` are separate counters** — vLLM itself distinguishes prefill from decode tokens; this is exactly the distinction the harness's `reported_tok_s` conflates (B3).
- Supporting: `vllm:num_requests_waiting`, `vllm:num_requests_running`, per-request `request_num_preemptions`.

### Part C references (availability verified)
- ≤1B rewriter candidates: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen3-0.6B`, `meta-llama/Llama-3.2-1B` (all HTTP 200). Local inference on the A100 — no external API needed (constraint: no API budget).

### Arithmetic already verified (will re-derive in submission)
- KV/token = 8 heads × 128 × 2 (K+V) × 28 layers × 2 B (fp16) = **114,688 B = 112 KiB/token**.
- `reported_tok_s = num_requests × (prompt_len + gen_len) / wall_clock_s` — exact on all 13 rows (spot-checked batch-24 long: 24×4096/61.16 = 1607.4 ✓).
- Honest goodput batch-24 long row: 24×512/61.16 = **200.9 tok/s** (both derivations agree).

## Architecture / approach decisions

- **Evidence-first workflow:** every A2/A3 claim gets a tiny standalone
  script under `partA/experiments/` whose output (command + numbers) is
  pasted into NOTEBOOK.md *as you go* — never write the notebook after the
  fact; the assignment explicitly grades dead ends.
- **Corpus:** read FLORES tarball directly with Python (no `datasets` dep);
  use `dev` (997) for measurement, keep `devtest` (1012) as a held-out
  cross-check (caveat signal for A1).
- **A3 matrix:** 4 tokenizers (gpt2, xlm-roberta-base, mt5-small,
  indic-bert) × 4 denominators (whitespace word, grapheme cluster via
  `regex \X`, UTF-8 byte, parallel sentence) × 5–8 languages.
- **B:** pure arithmetic + CSV analysis (no GPU, no vLLM run) — verify the
  `reported_tok_s` identity programmatically across all 13 rows.
- **C:** reasoned memo with labeled arithmetic; reviewer-throughput math is
  the binding constraint to surface (10 h/wk × 2 wks, 2 of 6 languages).

## Task List

### Phase 0: Foundation
- [ ] **Task 1: Environment setup**
  - `pip install transformers sentencepiece`; verify `tiktoken`, `regex`,
    `transformers` import; verify HF tokenizer load of `xlm-roberta-base`.
  - Acceptance: all imports succeed; one tokenizer encodes a Hindi sentence.
  - Verification: `python -c "import tiktoken, transformers, regex; ..."`.
  - Scope: XS (no repo files).

### Phase 1: Part A (50 pts — highest value, highest risk)
- [ ] **Task 2: A1 corpus build** (`partA/corpus/`, `partA/README.md`)
  - Download FLORES tarball; extract `dev` for ≥4 languages (eng, hin + 2
    Dravidian; recommend 5–8: + tel, tam, kan, ben, mar for Part C reuse);
    emit per-language files aligned by sentence id; document size, domain,
    preprocessing (NFC, strip, no other changes), and the **"what this
    corpus cannot tell you"** paragraph (wikinews domain, ~21-word
    sentences, translationese, no code/colloquial/product text, sample-size
    vs production traffic).
  - Acceptance: ≥4 parallel languages; README states size/domain/preproc/
    caveats; devtest reserved for cross-check.
  - Dependencies: Task 1.
- [ ] **Task 3: A2 flaw-isolation experiments** (`partA/experiments/exp_*.py`)
  Each experiment = exact command + before/after numbers + one-sentence
  causal statement. Planned experiments:
  - **E1 — seeded double spaces:** count `split(" ")` empty strings in both
    sample files; recompute fertility with `split()` vs `split(" ")`;
    report Δ direction/magnitude (per-line inflation: one empty word on the
    affected line → ~11% deflation on that line under macro-average).
  - **E2 — macro vs aggregate:** `mean(tokens_i/words_i)` vs
    `Σtokens/Σwords` on both corpora; show divergence (short lines get
    equal weight under macro).
  - **E3 — shared numerator (the conceptual flaw):** show
    `tok/char = (tok/word) / (chars/word)` is an algebraic transform —
    compute chars/word per language and demonstrate the "two metrics agree"
    is arithmetic coupling, not independent confirmation.
  - **E4 — denominator not constant across scripts:** code points per
    grapheme cluster and graphemes per whitespace-word per language
    (Devanagari combining matras vs Latin); shows tok/char partly measures
    Unicode density, not tokenizer quality.
  - **E5 — looks-suspicious-but-fine (flag as FINE, with evidence):**
    `random.seed(1337)` (grep: no RNG calls; results identical with/without
    seed → Δ = 0) and/or `unicodedata.normalize("NFC")` (measure Δ = ~0).
    Must explicitly state these are fine *because measured*.
  - **E6 (candidate) — `.lower()`:** measure with/without lowercasing on
    English (no-op for Devanagari); keep only if Δ meaningful — otherwise
    record as measured-fine in the notebook.
  - Acceptance: every claim in the eventual A2 write-up maps to an experiment
    with captured output; zero unverified claims.
  - Dependencies: Task 1.
- [ ] **Task 4: A3 corrected analysis** (`partA/fertility_corrected.py`)
  - Tokenizers: gpt2 + xlm-roberta-base + mt5-small (+ muril-base-cased if
    time; indic-bert is gated); denominators: word / grapheme cluster (`regex \X`) / UTF-8 byte /
    parallel sentence; corpus: A1 FLORES dev.
  - Output: full tokenizer × denominator × language table + the answer to
    "which single number drives routing-and-cost": **tokens per parallel
    sentence** (holds *meaning* constant), with the reasoning about what
    each denominator fails to hold constant.
  - Acceptance: ≥2 tokenizers × ≥2 denominators actually run; single-number
    answer justified in writing.
  - Dependencies: Tasks 1–2.
- [ ] **Task 5: A4 memo** (`partA/memo.md`, ≤1 page)
  - Corrected headline numbers, routing recommendation, biggest caveat, one
    production metric (e.g., per-language *token amplification ratio*
    measured online: tokens/request per language ÷ tokens/request English).
  - Dependencies: Task 4.

### Phase 2: Part B (20 pts — fast, independent; do early for momentum)
- [ ] **Task 6: B1–B3 calculations** (`partB/`)
  - B1: show KV/token arithmetic (114,688 B) + max-concurrent-sequence
    budget with explicit GB/GiB assumption sensitivity (~25–27), compare to
    log rows (kv_util 0.93@24 no-preempt, 0.97@32 w/ 7 preempted).
  - B2: identify anomaly (throughput 1607.4→1384.0→1298.5 while
    preempted_seqs 0→7→23, TTFT 500→637→955 ms, kv_util pinned 0.97);
    mechanism = KV exhaustion → preempt/recompute; one fix with predicted
    quantitative effect (e.g., `max_num_seqs=24` → 0 preemptions, TTFT
    ~500 ms, reported ~1600 vs 1384 = +16%; or fp8 KV → 56 KiB/token →
    ~51 seqs → batch 48 fits).
  - B3: programmatic identity check over all 13 rows; two independent
    derivations of 200.9 tok/s; what the report should have said.
  - Acceptance: identity verified on every row; both goodput derivations
    shown; no claim without a number.
  - Dependencies: none (can parallelize with Phase 1).
- [ ] **Task 7: B4 paragraph** (`partB/b4.md`)
  - Counter: `vllm:num_preemptions` (expect ≥7 at batch 32, ≥23 at 48, 0
    at ≤24) + `vllm:kv_cache_usage_perc` (expect ≥0.93 sustained); note
    prompt vs generation token counters for B3.
  - Dependencies: Task 6.

### Phase 3: Part C (15 pts)
- [ ] **Task 8: C decision memo** (`partC/memo.md`, ≤1 page)
  - Explicitly labeled: **assumptions** (baseline formality, reviewer
    10 h/wk covers only Hindi+Kannada = 2/6 languages, 3-week review);
    **arithmetic** (data volume: e.g., ~1k casualized pairs/lang × 6;
    synthesis + LoRA-SFT cost on 1×A100-80GB within 2 weeks; **reviewer
    throughput: 10 h/wk × 2 wks = 20 h ≈ 1,200 pairs at 1 min/pair →
    covers 2 languages only, Bengali/Marathi/Tamil/Telugu unreviewed →
    must state mitigation**); **success metric w/ numeric threshold**;
    **kill criterion w/ deadline**; **day-1 experiment** (build 100-sample
    eval set + run prompt-only baseline to measure the actual gap —
    cheapest signal first).
  - Rewrite candidates verified: Qwen2.5-0.5B-Instruct / Qwen3-0.6B
    (local inference, no API budget).
  - Dependencies: none.

### Phase 4: Submission packaging
- [ ] **Task 9: NOTEBOOK.md + AI_USAGE.md**
  - Chronological hypothesis→experiment→result→revision log compiled *from
    work already done* (include dead ends: e.g., gated FLORES on HF →
    fbaipublicfiles tarball; Aksharagarbha nonexistent; websearch 402).
  - AI_USAGE.md: honest account (which scripts AI wrote, which claims AI
    got wrong and how caught).
  - Dependencies: Tasks 2–8.
- [ ] **Task 10: Defense rehearsal + packaging**
  - Rehearse live re-derivation of: 114,688 B/token, 200.9 tok/s (both
    ways), one fertility experiment, the A3 denominator argument.
  - Sanity-check repo structure matches `your-submission/` spec.
  - Dependencies: all.

## Checkpoints
- **After Tasks 1–2:** corpus loads, ≥4 languages aligned, tokenizer loads.
- **After Tasks 3–4:** ≥4 experiments with captured outputs; A3 table runs
  end-to-end. (Review: do all A2 claims have numbers? If not, don't proceed.)
- **After Tasks 6–7:** B arithmetic independently re-verified.
- **Before Task 10:** every claim in every memo traces to a file in
  `partA/experiments/` or `partB/` — this is the defense gate.

## Risks and Mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| Web search backend down (HTTP 402) | Medium | Primary sources fetched directly (HF API, raw GitHub, docs.vllm.ai) — all facts above verified this way; cite them |
| HF `facebook/flores` gated | Low | Public 25.6 MB tarball at fbaipublicfiles (verified reachable) |
| Claiming a flaw without measurement (−5 each) | High | Rule: no sentence in A2 write-up without an `exp_*.py` output; flag "fine" items *with* Δ=0 evidence |
| Retroactive-looking notebook | High | Log every experiment in NOTEBOOK.md as it runs, dead ends included |
| `transformers`/SentencePiece install issues on py3.13 | Low | Fallback tokenizer pair: tiktoken gpt2 + cl100k/o200k (both multilingual-ish) — but prefer XLM-R/mt5 as the Indic-aware requirement |
| GB vs GiB ambiguity in B1 | Low | Show both, state assumption, compare against log to pick the consistent one |
| Defense can't re-derive a number | Gate fail | Task 10 rehearsal covers the 4 canonical numbers + one live experiment |

## Open Questions
- None blocking. (Optional: whether to also submit `devtest` cross-check
  numbers — cheap and reads as rigor; do it if time permits.)
