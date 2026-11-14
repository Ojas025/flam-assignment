# Task C

## Decision memo

**Scenario**: the product team wants our assistant's replies to sound casual and conversational in Hindi, Kannada, Tamil, Telugu, Bengali and Marathi — current outputs are too formal/textbook. 
Three paths are on the table:
- (a) an SFT pass on synthetic "casualized" response pairs
- (b) a small (≤1B) inference-time rewriter model after the main model
- (c) prompt-engineering only

**Constraints**: one A100-80GB for 2 weeks, one native-speaker reviewer
(Hindi + Kannada only) for 10 h/week, launch review in 3 weeks, no
external API budget.

---

### Assumptions made
- The main model is FLM-4B-Instruct from `model_spec.md` (4.2B dense), we own the weights.
- All synthetic data comes from our own models, since there is no API budget. Quality in Tamil, Telugu, Bengali and Marathi is unknown.
- Training examples are (user prompt/context → casual response). The original formal response is retained for comparison and semantic auditing, not used as the SFT target.
- No reviewer exists for ta/te/bn/mr, so I make no quality claim for them. Hindi/Kannada review does not transfer, and spoken Tamil/Telugu differ a lot from written forms.
- Use BF16/FP16 base weights and LoRA rather than full-parameter fine-tuning.

---

### Back-of-envelope arithmetic

Training
- LoRA is the safer choice on an 80GB A100, primarily for memory headroom, not because it is guaranteed to train faster.
- Model memory: 4.2B * 2 =8.4GB of FP16/BF16 weights. This is comfortably below the A100-80GB capacity.
- Generate 2000 synthetic pairs per language -> 12,000 pairs.
- At an average of 600 tokens/pair (since indic tokenization is costly) -> 7.2M tokens per epoch
- Trained for 3 epochs -> 7.2M * 3 -> 21.6M tokens at 
- GPU budget -> 336 hours available
- Total model parameters = 4.2 * 1e9
	LoRA FLOPs per token:
	- Forward Pass -> 2P
	- Backward Pass -> 2P
	- Gradient checkpointing (optional) -> 2P
- Total FLOPs -> 6 * 4.2 * 1e9 -> `5.44 * 1e17`
- A100 peak BF16 dense compute -> 312 TFLOPs
- Model Flops Utilization (MFU) -> ~25%
- Time = FLOPs / (peak * MFU) = `1.94 hours`, so 1-2 hours per run.

Prompt
- Few-short prompt per request -> ~1k tokens
- A 1,000-token prompt would consume approximately 109 MiB of KV cache per active request at 114,688 bytes/token, but a 100-token instruction is only about 11 MiB.
- If a short-style instruction is sufficient, this cost will be dropped.

Reviewer
- 10 h/week × 3 weeks = 30 h. 
- Pairwise ≈ 45 s (80/h); audit-and-fix ≈ 2 min (30/h)
- Max ~1600 judgements -> 13.3% of the data.

The scarce resource / bottleneck, is reviewer time and not GPU.	

---

### Success metric with a numeric threshold
On a held-out, blinded Hindi and Kannada evaluation set, require:
- At least 70% preference for the adapted model over the original on naturalness and conversational tone.
- At least 95% semantic preservation, with no material factual additions, omissions or changes.
- No more than a 2-percentage-point regression on the existing general evaluation suite.

Launch scope is Hindi and Kannada only. The other four ship only with a flag labelled unverified, if must.

---

### Path choices
- Primary choice -> LoRA SFT, bakes the style into the weights. It removes the ~1k-token few-shot prompt per request
- Rewriter: A 1B FP16 model needs roughly 2GB for weights, so memory is not the issue. It adds a serial generation pass to every response, increases latency and creates another opportunity to alter facts, names or numbers. That is unnecessary when the main model can be adapted, and we are already constrained by reviewer time.
- Prompt-only: It is cheap and reversible, so use it as the baseline. But, prompts alone may not provide a reliable behavioral change.

---

### Kill criterion
- By day 7, audit 200 synthetic examples per reviewed language. 
- Abandon SFT for either language if more than 15% fail the rubric through unnatural register, grammatical problems or semantic drift, or if material meaning errors exceed 5%. - Use prompt-only for that language only if it independently meets the success criteria; otherwise keep the existing behaviour.

---

### First experiment I would run on Day 1

Generate casual prompt with 5 reviewer-fixed examples per language, spanning factual questions, explanations, instructions and short conversational turns.
Run it against the model on 60 generated held-out prompts per language.
The reviewer judges the blind pairs for Hindi and Kannada, scoring conversational naturalness and meaning preservation -> (60 + 60) * 45s

Decision points:
- If >= 65% win and <=5% meaning error -> can ship with proper Prompt Engineering, SFT run is optional.
- 50-65% -> proceed with the SFT run, using outputs from Prompt Engineering as the synthetic teacher data samples.
- <=50% -> The prompts are bad. Raise the issue, and fix it now.
