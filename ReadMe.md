# HMM Solver

HMM Solver is a small web application for experimenting with Hidden Markov Models (HMMs). It combines a Next.js frontend with Python implementations of the core HMM algorithms so you can inspect model parameters, compute probabilities, and estimate hidden state sequences interactively.

## What this project contains

This project includes:

- A discrete HMM calculator for manually entered parameters
- A continuous HMM calculator for Gaussian emission models
- A conversion workflow from continuous Gaussian HMMs into discrete observation models
- Results for the main HMM inference and learning algorithms

## HMM modes in the app

### 1. Discrete HMM

The discrete tab lets you define:

- hidden states
- observation symbols
- initial distribution $\pi$
- transition matrix $A$
- emission matrix $B$

You can then run:

- Forward algorithm
- Backward algorithm
- Viterbi algorithm
- Baum-Welch algorithm

The results are displayed as probability values, state paths, and parameter updates.

### 2. Continuous HMM

The continuous tab supports Gaussian emission HMMs:

- 1D continuous observations
- multi-dimensional Gaussian support
- parameter input for means and variances/covariances
- conversion to a discretized HMM for comparison and experimentation

This is useful for modeling real-valued data where observations are not categorical symbols.

## Implemented algorithms

The solver supports the classic HMM tasks:

- Forward algorithm: computes the probability of an observation sequence
- Backward algorithm: computes backward probabilities for a sequence
- Viterbi algorithm: finds the most likely hidden-state path
- Baum-Welch algorithm: re-estimates model parameters from observed sequences via EM

## Project structure

```text
hmm-solver/
├── app/
│   ├── api/hmm/route.ts          # API endpoint that forwards requests to Python
│   └── page.tsx                  # Main UI and tab-based HMM interface
├── components/
│   ├── DemoSection.tsx           # Ambiguous phrase demo UI
│   ├── ParameterInput.tsx        # Parameter editing forms
│   ├── ResultsDisplay.tsx        # Results rendering
│   └── AlgorithmFormulas.tsx     # Formula panel
├── lib/
│   ├── hmm_algorithms.py         # Core HMM logic and algorithm implementations
│   ├── hmm_executor.py           # Python bridge that executes the requested algorithm
│   └── utils.ts                  # Utility code
├── package.json                  # Next.js app configuration and scripts
├── pnpm-lock.yaml                # Dependency lock file
├── ReadMe.md                     # Project documentation
└── next.config.mjs               # Next.js configuration
```

## How the app works

The frontend sends JSON requests to the route in `app/api/hmm/route.ts`. That API then launches the Python executor, which passes the request into the HMM logic in `lib/hmm_algorithms.py`.

The Python layer is where the model math lives:

- discrete HMM calculations
- continuous Gaussian HMM calculations
- Viterbi decoding
- parameter estimation
- conversion between continuous and discrete representations

## Usage

### Prerequisites

- Node.js and pnpm
- Python available on the PATH as `py` on Windows or `python3` on Unix-like systems

### Run locally

From the project folder:

```bash
cd hmm-solver
pnpm install
pnpm dev
```

Then open:

```text
http://localhost:3000
```

### Using the interface

1. Open the app in the browser.
2. Choose a tab:
   - Ambiguous Phrase
   - Discrete HMM
   - Continuous HMM
3. Edit the hidden states and observation vocabulary.
4. Select the algorithms you want to evaluate.
5. Enter the HMM parameters or use the demo example.
6. Run the computation and inspect the results panel.

### Example workflow

For a discrete HMM:

- set states such as `S1`, `S2`, `S3`
- set observation symbols such as `o1`, `o2`, `o3`
- enter $\pi$, $A$, and $B$
- choose forward, backward, Viterbi, or Baum-Welch
- click the run action to calculate the outputs

For a continuous HMM:

- choose 1D or multi-dimensional observations
- provide means and variances/covariances
- run the selected algorithm
- optionally convert continuous parameters into a discrete-symbol model

## Notes

- The demo is designed to illustrate ambiguous tagging behavior in a compact example.
- For out-of-vocabulary words in the demo, the implementation defaults to a fallback index, so results may be less reliable if the word is not in the learned vocabulary.
- The main math and HMM algorithms are implemented in `lib/hmm_algorithms.py`.

## Summary

This app is meant as an educational and interactive HMM playground. It helps you understand the roles of $\pi$, $A$, and $B$, test classic algorithms, and compare discrete and continuous model behavior in a simple interface.
