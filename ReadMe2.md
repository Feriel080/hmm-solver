## Practical Work: Resolving Grammatical Ambiguity with HMM and spaCy

### Context

In natural language processing (NLP), a word may have several possible grammatical categories (for example, *brise* can be a noun or a verb). To resolve this ambiguity, probabilistic models such as Hidden Markov Models (HMM) or neural tools like spaCy are used.

### Objectives

1. Use the spaCy library to perform tokenization and morphosyntactic analysis of a sentence.
2. Identify the tokens of the sentence:  *"la petite brise la glace"* .
3. Estimate a Hidden Markov Model (HMM) from a small annotated corpus, then apply the Viterbi algorithm to determine the most likely grammatical tagging of the target sentence.
