# TASK.md — The Audit (Flam AI Team Intern Assignment)

## What this is

A take-home assignment from Flam (Bangalore) for an AI team intern screening:
**"The Audit"** — designed for ~10 focused hours over 5 days, followed by a
30-minute live defense session. Graded on thinking-per-hour, not volume.

## The situation

A previous intern benchmarked the company's tokenizer and serving stack,
wrote `REPORT_v0.md`, and left. Leadership is **about to make routing and
capacity decisions based on that report**. Your job is to audit the report
before that happens.

The report is deliberately written with confident, wrong conclusions (e.g.
*"No further measurement needed — the two metrics agree, so the result is
robust"*). You are scored for catching the flaws **with evidence**.

So this is not "build a system" — it is **adversarial peer review of
finished-looking work that contains planted flaws, with real decision stakes
attached.**

## Ground rules

1. **The evidence rule.** Every claim you make ("this line is a bug", "this
   metric is misleading") must ship with a minimal experiment: the exact
   command, the before/after numbers, and one sentence on why the delta
   proves your claim. A claimed flaw without measured evidence scores
   **−5 each** (Part A2 floor at 0). Confident hallucination is punished
   harder than being wrong.
2. **The defense.** In the live 30-minute session you will re-derive your
   numbers, modify your code on the spot, and answer counterfactuals.
   Anything in your submission is fair game. **Fabricated evidence =
   automatic fail**, regardless of score.

AI tools are explicitly allowed and expected. The assignment is engineered
so that AI without your own understanding produces a mediocre submission.

## Inputs (in `resources/starter_kit/`)

| File | What it is |
|---|---|
| `fertility.py` | The script that produced the tokenizer numbers — has code bugs, ≥1 conceptual flaw, and ≥1 thing that looks suspicious but is actually fine |
| `REPORT_v0.md` | The intern's report and its (wrong) recommendations |
| `corpus_sample/` | Tiny parallel English + Hindi corpora (10 sentences each, smoke-test toys with seeded double spaces) |
| `bench/model_spec.md` | Serving setup: FLM-4B-Instruct (28 layers, d_model 3072, 24 Q / 8 KV heads GQA, head_dim 128, fp16) on 1× NVIDIA L4 (24 GB), max_model_len 4096, gpu_memory_utilization 0.92, ~1.6 GB non-KV overhead |
| `bench/bench_log.csv` | 13-row load-test log: a short-prompt sweep (512/256, batch 1→64) and a long-context sweep (3584/512, batch 4→48) |

## The three parts

### Part A — Tokenizer audit (50 pts)

Underlying concept: a subword tokenizer (BPE) splits text using statistics
from its training data. **Fertility = tokens per word.** GPT-2's BPE was
trained mostly on English, so Indic scripts explode into many tokens per
word. The report claims *"Hindi is 5.89× worse → serving Hindi costs ~6×
more → route Indic traffic to a special tokenizer."*

- **A1 (10 pts) — Build a real eval corpus.** At least 4 languages:
  English, Hindi, and two Dravidian languages (Kannada / Tamil / Telugu /
  Malayalam). FLORES-200 is the suggested source (parallel corpora hold
  *meaning* constant across languages). Document size, domain,
  preprocessing, and **one paragraph on what your corpus cannot tell you**
  (caveats are graded as signal, not weakness).
- **A2 (20 pts) — Audit the script and the metric.** Find the code bugs,
  the conceptual flaw (the code computes exactly what it says, but what it
  says is the wrong thing to measure), and correctly *not* flag the
  harmless-looking thing. Isolate every claimed flaw, measure its effect,
  and state direction + magnitude. Known planted flaws include:
  - `line.split(" ")` — seeded double spaces create empty strings that
    inflate the word denominator, deflating fertility for the affected lines;
  - macro-averaging per-line ratios instead of aggregating tokens / Σwords;
  - the shared numerator: tok/word and tok/char are correlated, not
    independent confirmations, so "two metrics agree → robust" is bogus;
  - the conceptual denominator problem: a whitespace word and a code point
    are not constant units across scripts (combining matras, conjuncts).
- **A3 (12 pts) — Corrected analysis.** Recompute with ≥2 tokenizers (one
  multilingual/Indic-aware) × ≥2 denominators (per word, per grapheme
  cluster, per UTF-8 byte, per parallel sentence). Then answer: **which
  single number should drive a routing-and-cost decision, and why?** Hint:
  think about what the denominator is supposed to hold constant across
  languages (answer direction: tokens per *parallel sentence* — equal
  meaning in every language).
- **A4 (8 pts) — Recommendation memo (≤1 page).** Corrected headline
  numbers, routing recommendation, biggest caveat, and the one production
  metric that would catch this analysis being wrong.

### Part B — Capacity reconciliation (20 pts)

- **B1 (7 pts)** — From the spec alone: (a) KV-cache bytes per token,
  exactly (8 KV heads × 128 head_dim × 2 (K+V) × 28 layers × 2 bytes fp16 =
  **114,688 bytes ≈ 112 KiB/token**); (b) approximate max concurrent
  4096-token sequences this GPU can hold (~25–27 after weights + overhead).
  Show arithmetic, then check against the log.
- **B2 (6 pts)** — The long-context sweep has a throughput anomaly: it
  rises to batch 24 (1607.4 tok/s) then **falls** at batch 32 (1384.0) and
  48 (1298.5), against naive "throughput scales with batch" intuition.
  Identify it, explain the mechanism using specific rows/columns (KV
  exhaustion → scheduler preemption/recompute → TTFT balloons 500→637→955
  ms while ITL plateaus ~100 ms; `preempted_seqs` = 0/7/23), and propose
  one config/deployment change with a predicted quantitative effect.
- **B3 (4 pts)** — Both of REPORT_v0 Section 2's conclusions come from
  misreading one column: `reported_tok_s`. The identity is
  `num_requests × (prompt_len + gen_len) / wall_clock_s`, so it **counts
  prefill (prompt) tokens alongside output tokens** — that's why long
  prompts "look faster" and why "batch 48 → ~3200 tok/s" is fantasy.
  Derive the honest goodput of the batch-24 long-prompt row two independent
  ways (both give **200.9 tok/s**), and say what the report should have
  said.
- **B4 (3 pts)** — One paragraph: which single serving-stack counter
  confirms the B2 mechanism, and what value do you expect it to show?

### Part C — Decision memo (15 pts)

Product wants **casual (not textbook-formal)** replies in Hindi, Kannada,
Tamil, Telugu, Bengali, Marathi. Three paths:

- (a) SFT on synthetic "casualized" response pairs
- (b) a ≤1B inference-time rewriter model after the main model
- (c) prompt-engineering only

**Constraints:** one A100-80GB for 2 weeks · one native-speaker reviewer
(**Hindi + Kannada only**) for 10 h/week — the scarcest resource, covering
only 2 of 6 languages · launch review in 3 weeks · no external API budget.

Write a ≤1-page recommendation with explicitly labelled: **assumptions**,
**back-of-envelope arithmetic** (data volume, training/serving cost,
reviewer throughput), a **success metric with a numeric threshold**, a
**kill criterion** (the observation that would make you abandon your path,
and by when), and the **day-1 first experiment**. There is no single right
answer; there are many wrong ways to reason.

## Deliverables

```
your-submission/
  NOTEBOOK.md     # graded, chronological: hypothesis → experiment → result → revision, including dead ends
  AI_USAGE.md     # honest account of where AI helped / misled you
  partA/          # corpus prep scripts + corrected analysis code
  partB/          # calculations + written answers
  partC/memo.md
```

`NOTEBOOK.md` is graded, not decorative: *"a notebook with no dead ends was
written after the fact."*

## Scoring

| Component | Pts |
|---|---|
| A1 corpus construction & caveats | 10 |
| A2 script/metric audit (evidence rule applies) | 20 |
| A3 corrected analysis & denominator reasoning | 12 |
| A4 recommendation memo | 8 |
| B1–B4 capacity reconciliation | 20 |
| C decision memo | 15 |
| Defense (a gate — fail it and the score doesn't save you) | 15 |
| **Total** | **100** |

- Unverified claimed flaws: **−5 each** (A2 floor at 0).
- Fabricated evidence: **fail**.
- Closing guidance: *"we'd rather see three claims you can defend to the
  death than ten you can't."*

## The one-sentence essence

**You are being hired to be the person who stops bad numbers from reaching
leadership** — the assignment tests whether you can build trustworthy
measurements, distinguish real bugs from cosmetic red flags and prove each
claim empirically, see through metrics that answer the wrong question, and
convert corrected numbers into a decision under tight resource constraints —
while being honest about AI's role and able to defend every number live.
