# TODO — The Audit submission

> Plan details + verified research: [`plan.md`](plan.md). Full task brief: [`../TASK.md`](../TASK.md).

## Daily schedule (5 sessions, Tue Nov 10 → Sat Nov 14 — hard deadline Sat)

| Day | Date | Session | Est. |
|---|---|---|---|
| 1 | Tue Nov 10 | Task 2 — A1: FLORES download + parallel corpus + caveats | 1.5 h |
| 2 | Wed Nov 11 | Task 3 — A2: all flaw-isolation experiments (the 20-pointer) | 3 h |
| 3 | Thu Nov 12 | Tasks 4+5 — A3 corrected analysis + A4 memo | 2.5 h |
| 4 | Fri Nov 13 | Tasks 6+7+8 — Part B (B1–B4) + Part C memo | 3.5 h |
| 5 | Sat Nov 14 | Tasks 9+10 — NOTEBOOK/AI_USAGE final pass + defense rehearsal + package (rest of day = slack for anything weak) | 1.5 h + slack |

**Session protocol (proper understanding, every day):**
1. Brief the task: goal + evidence requirements
2. **You write/run**; agent answers questions, checks your reasoning, runs read-only verification on request (coach-&-verify mode)
3. You explain back the day's key causal sentence *in your own words* (that's the defense rehearsal)
4. Log to `NOTEBOOK.md` immediately — hypothesis → experiment → result → revision, dead ends included (never after the fact)
5. Check off todo + commit (save point)

NOTEBOOK logging is **embedded daily**, not a Day-5 task.

## Phase 0: Foundation
- [x] Task 1: Environment setup via uv (2026-11-10) — `uv sync` installs tiktoken/transformers/sentencepiece/regex/protobuf; verified: gpt2=23, xlm-roberta-base=3, mt5-small=6, muril-base-cased=2 tokens for "नमस्ते दुनिया". ⚠️ `ai4bharat/indic-bert` is GATED (401) → use `google/muril-base-cased` as the Indic-specific tokenizer

## Phase 1: Part A (50 pts — do first, highest value/risk)
- [ ] Task 2 (A1): Download FLORES-200 public tarball (25.6 MB, no login) → build ≥4-lang parallel corpus from `dev` split; document size/domain/preprocessing + "what this cannot tell you" paragraph; reserve `devtest` for cross-check → `partA/corpus/`, `partA/README.md`
- [ ] Task 3 (A2): Run flaw-isolation experiments, each with command + before/after + causal sentence, logged in NOTEBOOK.md as they run:
  - [ ] E1: double-space `split(" ")` vs `split()` on sample corpora (direction + magnitude)
  - [ ] E2: macro-average vs aggregate (Σtok/Σwords)
  - [ ] E3: shared numerator — tok/char is algebraic transform of tok/word (coupling, not confirmation)
  - [ ] E4: denominator not constant — codepoints/grapheme, graphemes/word per language
  - [ ] E5: flag `random.seed(1337)` and/or NFC as FINE **with Δ=0 evidence**
  - [ ] E6 (optional): `.lower()` effect on English
- [ ] Task 4 (A3): `partA/fertility_corrected.py` — ≥2 tokenizers (gpt2 + xlm-roberta-base + mt5-small) × ≥2 denominators (word / grapheme `regex \X` / byte / parallel sentence) on FLORES; answer: single number = **tokens per parallel sentence** + why → `partA/memo.md` inputs
- [ ] Task 5 (A4): ≤1-page memo — corrected headlines, routing rec, biggest caveat, 1 production metric → `partA/memo.md`

## Phase 2: Part B (20 pts — fast; can run parallel to Phase 1)
- [ ] Task 6 (B1–B3): `partB/` — KV/token arithmetic (114,688 B) + max-seq budget (~25–27, GB/GiB sensitivity); B2 anomaly (1607→1384→1299, preempted 0→7→23) + fix w/ predicted effect; B3 identity check on all 13 rows + both derivations of 200.9 tok/s
- [ ] Task 7 (B4): `partB/b4.md` — `vllm:num_preemptions` (expect ≥7 @32, ≥23 @48, 0 ≤24) + `vllm:kv_cache_usage_perc` (≥0.93)

## Phase 3: Part C (15 pts)
- [ ] Task 8 (C): `partC/memo.md` ≤1 page — labeled assumptions / arithmetic (incl. reviewer throughput: 20 h ≈ 1,200 pairs, only 2 of 6 languages covered) / success metric w/ number / kill criterion w/ deadline / day-1 experiment

## Phase 4: Packaging
- [ ] Task 9: `NOTEBOOK.md` (chronological, dead ends included — write as you go!) + `AI_USAGE.md` (honest)
- [ ] Task 10: Defense rehearsal — re-derive 114,688 B, 200.9 tok/s ×2 ways, one fertility experiment, A3 denominator argument; verify `your-submission/` structure

## Checkpoints
- [ ] **After Tasks 1–2:** corpus aligned ≥4 langs, tokenizer loads
- [ ] **After Tasks 3–4:** every A2 claim has captured numbers; A3 table runs end-to-end
- [ ] **After Tasks 6–7:** B arithmetic re-verified programmatically
- [ ] **Before Task 10:** every claim in every memo traces to an experiment file (defense gate)
