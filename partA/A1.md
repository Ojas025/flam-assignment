# EVAL_CORPUS.md — the A1 eval corpus

The corpus we measure everything in Part A on: FLORES-200, six languages,
two splits.

## Source and size

Source: [FLORES-200](https://github.com/facebookresearch/flores), downloaded
with `partA/corpus/download_corpus.py` (SHA-256 checked). It is a *parallel*
corpus — every language holds the same meaning sentence by sentence, in the
same order, which is what makes cross-lingual tokenizer numbers comparable.

| Split | Sentences per language | Languages | Language-sentence instances |
|---|---|---|---|
| dev | 997 | 6 | 5,982 |
| devtest | 1,012 | 6 | 6,072 |
| **Total** | **2,009** | **6** | **12,054** |

Languages: English, Hindi, Kannada, Tamil, Telugu, Malayalam.

Note on the total: **12,054 is language-sentence instances, not unique
sentences.** There are 2,009 unique source sentences, each appearing once
per language (2,009 × 6 = 12,054).

## Domain

Written reference prose, roughly a third each of news stories, how-to and
reference-book chapters, and travel-guide articles (the metadata bundled
with the dataset says Wikinews, Wikibooks, and Wikivoyage — about 35%, 30%,
and 35% of the dev split). The text is full sentences with complete
punctuation, standard spelling, and a formal register: the kind of language
you find in an encyclopedia or a guidebook, not the kind you find in a chat
window.

## Preprocessing

Run by `partA/corpus/preprocess.py`:

- **Unicode NFC normalization** — Indic text has more than one byte
  representation for the same visible character; NFC makes each character
  one canonical form.
- **Whitespace cleanup** — leading/trailing whitespace stripped, runs of
  spaces/newlines collapsed to a single space, empty lines dropped.
- **Parallel alignment checked across all six languages** — the pipeline
  refuses to write output if any language's line count differs from the
  others, so line *N* is the same sentence in every file.
- **Nothing else.** Casing, wording, and punctuation were preserved exactly;
  preprocessing cleaned the bytes, it did not rewrite the text.

Reproduce:

```bash
python partA/corpus/download_corpus.py --corpus flores
python partA/corpus/preprocess.py --corpus flores
```

Output lands in `partA/corpus/preprocessed_corpus/flores/{dev,devtest}/<lang>.txt`.

## What this corpus cannot tell you

FLORES tells us how our tokenizers treat *written* text, but the product
needs something else: casual, chatty assistant replies. This corpus has
none of that — no code-switching, no Indic words typed in Latin script, no
emoji, no abbreviations, no half-finished sentences. A tokenizer that looks
cheap on travel-guide Hindi may look expensive on chatty Hindi, and we won't
see that here. Size is the second limit: ~2,000 sentences per language gives
a stable average, but it's far too small to say anything about rare text —
long names, URLs, numbers, mixed scripts — which is exactly where token
costs blow up. So treat these numbers as a solid cross-lingual comparison
under equal meaning, not a forecast of production traffic.
