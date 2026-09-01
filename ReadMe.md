# HMM Solver

## Features

The project covers two main variations of HMMs:

* **Discrete HMM**
* **Continuous HMM** *(currently supports only the 1-dimensional case – multi-dimensional case is under development)*

## Implemented Algorithms

* **Forward algorithm** – computes the probability of an observed sequence
* **Backward algorithm** – used in combination with forward for parameter estimation
* **Viterbi algorithm** – finds the most likely sequence of hidden states
* **Baum-Welch algorithm** – learns HMM parametes from observation sequences (expectation-maximization)

## Usage

```bash
# Install and run (already set up)
pnpm install
pnpm dev

# Open browser
# → http://localhost:3000
```

## _HMM algorithms are implemented on [hmm_algorithms.py](https://github.com/Feriel080/hmm-solver/blob/main/lib/hmm_algorithms.py)_
