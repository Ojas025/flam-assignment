# Task C — decision memo

## Scenario

Make assistant replies sound casual in Hindi, Kannada, Tamil, Telugu, Bengali, and Marathi (today’s tone is too formal).

**Options**

- (a) LoRA SFT on synthetic casual pairs
- (b) Small (≤1B) rewriter at inference time
- (c) Prompt engineering only

**Constraints**

- 1× A100-80GB for 2 weeks
- One native reviewer (Hindi + Kannada only), 10 h/week
- Launch review in 3 weeks
- No external API budget

## Assumptions

- Base model: FLM-4B-Instruct (`model_spec.md`), 4.2B dense; we own the weights.
- Synthetic data from our own models only. Quality for ta/te/bn/mr is unknown.
- Training target: (prompt → casual reply). Keep the formal reply for audits, not as the SFT label.
- No reviewer for ta/te/bn/mr — no quality claim there. Hindi/Kannada review does not transfer.
- Train with BF16/FP16 + LoRA (not full fine-tune).

## Arithmetic

### Training

- LoRA preferred for memory headroom on the A100 (not because it is always faster).
- Weights: 4.2B × 2 ≈ **8.4 GB** — fits easily.
- Data: 2,000 pairs × 6 languages = **12,000** pairs.
- ~600 tokens/pair → **7.2M** tokens/epoch × 3 epochs = **21.6M** tokens.
- GPU budget: **336** hours.
- FLOPs per token: forward **2P**, backward **4P**; +**2P** if gradient checkpointing.
  - Without checkpointing: `6 × 4.2e9 × 21.6e6` → **5.44×10¹⁷** → **~1.94 h**
  - With checkpointing: `8 × 4.2e9 × 21.6e6` → **7.26×10¹⁷** → **~2.58 h**
- Peak 312 TFLOP/s × ~25% MFU → plan **2–3 hours** per run.

### Prompt cost

- Few-shot style prompt ~1k tokens ≈ **109 MiB** KV per active request.
- Short instruction ~100 tokens ≈ **11 MiB**. Prefer the short form if it works.

### Reviewer

- Capacity: 10 h/week × 3 weeks = **30 h**.
- Pairwise ~45 s (80/h); audit ~2 min (30/h).
- 1,200 pairwise + 400 audits = **15 + 13.3 = 28.3 h** → ~**1,600** judgements in 30 h (**13.3%** of the data).

**Bottleneck: reviewer time, not GPU.**

## Success criteria (Hindi + Kannada, blinded)

- ≥ **70%** preference on naturalness / conversational tone
- ≥ **95%** semantic preservation (no material fact changes)
- ≤ **2** pp regression on the general eval suite

Launch Hindi and Kannada only. Other four ship as unverified if required.

## Path choice

- **Primary: LoRA SFT** — style in the weights; drops the ~1k-token few-shot prompt.
- **Rewriter:** ~2 GB weights is fine, but adds a serial pass, latency, and another place to corrupt facts. Not worth it under reviewer scarcity.
- **Prompt-only:** cheap baseline; may not hold a reliable style change alone.

## Kill criterion

- By day 7, audit **150** synthetic examples per reviewed language (**2 × 150 × 2 min = 10 h** — week-1 reviewer capacity).
- Drop SFT for a language if >**15%** fail the rubric, or meaning errors >**5%**.
- Fall back to prompt-only only if it independently meets success criteria; otherwise keep current behaviour.

## Day-1 experiment

- 5 reviewer-fixed casual examples per language (mixed task types).
- Run on **60** held-out prompts per language.
- Blind pairwise review (Hindi + Kannada): naturalness + meaning → (60+60) × 45 s.

**Decide**

- ≥65% win and ≤5% meaning error → prompt engineering may ship; SFT optional
- 50–65% → run SFT, using prompt outputs as teacher data
- ≤50% → prompts are bad; fix before anything else
