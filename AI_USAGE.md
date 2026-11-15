# AI Usage

## How I Used AI

- Wrote most of the experiment + plotting code using AI, based on the research and  decisions of implementations that i did independently, this made scaffolding code faster - only for the fact that i knew what was to be done and this wasn't a blindly written code.
- An incident of the agent forging a sample set for the sake of displaying a bigger delta, which led me to misinterpret the experiment result initially
- I used the models in Opencode, wrote lots of duplicated code, unnecessary helper functions --- had to review and clean a lot of it.
- For the `split() vs split(" ")` experiment, the agent confidently insisted on keeping whitespace normalization out of preprocessing for the sake of showing the delta in experiment results.
- AI hardcoded `0.0` for the macro-vs-micro `tps_delta_pct` in `exp_macro_micro.py` instead of computing it, had to fix that
- For reports in Task B, all the calculations were done manually for better understanding, writing the findings and resulting report was done using AI.
