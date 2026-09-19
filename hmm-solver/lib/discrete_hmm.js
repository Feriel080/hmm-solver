function forwardAlgorithm(obs, Pi, A, B, states, vocab) {
    const wordToIdx = {};
    vocab.forEach((word, i) => { wordToIdx[word] = i; });
    const obsIdx = obs.map(word => wordToIdx[word] ?? 0);

    const T = obs.length;
    const N = states.length;

    // alpha[t][s] = probability
    const alpha = Array.from({ length: T }, () => Array(N).fill(0.0));

    // Initialization
    for (let s = 0; s < N; s++) {
        alpha[0][s] = Pi[s] * B[s][obsIdx[0]];
    }

    // Recursion
    for (let t = 1; t < T; t++) {
        for (let s = 0; s < N; s++) {
            for (let prevS = 0; prevS < N; prevS++) {
                alpha[t][s] += alpha[t - 1][prevS] * A[prevS][s] * B[s][obsIdx[t]];
            }
        }
    }

    // Total Probability
    const prob = alpha[T - 1].reduce((a, b) => a + b, 0);

    return {
        observation_sequence: obs,
        probability: prob,
        alpha_table: alpha,
        states: states,
    };
}

function backwardAlgorithm(obs, Pi, A, B, states, vocab) {
    const wordToIdx = {};
    vocab.forEach((word, i) => { wordToIdx[word] = i; });
    const obsIdx = obs.map(word => wordToIdx[word] ?? 0);

    const T = obs.length;
    const N = states.length;

    // beta[t][s]
    const beta = Array.from({ length: T }, () => Array(N).fill(0.0));

    // Initialization
    for (let s = 0; s < N; s++) {
        beta[T - 1][s] = 1.0;
    }

    // Recursion (backwards)
    for (let t = T - 2; t >= 0; t--) {
        for (let s = 0; s < N; s++) {
            for (let nextS = 0; nextS < N; nextS++) {
                beta[t][s] += A[s][nextS] * B[nextS][obsIdx[t + 1]] * beta[t + 1][nextS];
            }
        }
    }

    // Total probability
    let prob = 0.0;
    for (let s = 0; s < N; s++) {
        prob += Pi[s] * B[s][obsIdx[0]] * beta[0][s];
    }

    return {
        observation_sequence: obs,
        probability: prob,
        beta_table: beta,
        states: states,
    };
}

function viterbiAlgorithm(obs, Pi, A, B, states, vocab) {
    const wordToIdx = {};
    vocab.forEach((word, i) => { wordToIdx[word] = i; });
    const obsIdx = obs.map(word => wordToIdx[word] ?? 0);

    const T = obs.length;
    const N = states.length;

    // Viterbi variables
    const V = Array.from({ length: T }, () => Array(N).fill(0.0));
    const bp = Array.from({ length: T }, () => Array(N).fill(0));

    // Initialization
    for (let s = 0; s < N; s++) {
        V[0][s] = Pi[s] * B[s][obsIdx[0]];
        bp[0][s] = 0;
    }

    // Recursion
    for (let t = 1; t < T; t++) {
        for (let s = 0; s < N; s++) {
            let maxProb = 0.0;
            let maxState = 0;
            for (let prevS = 0; prevS < N; prevS++) {
                const prob = V[t - 1][prevS] * A[prevS][s] * B[s][obsIdx[t]];
                if (prob > maxProb) {
                    maxProb = prob;
                    maxState = prevS;
                }
            }
            V[t][s] = maxProb;
            bp[t][s] = maxState;
        }
    }

    // Backtracking
    const bestPathProb = Math.max(...V[T - 1]);
    const bestLastState = V[T - 1].indexOf(bestPathProb);

    const path = Array(T).fill(0);
    path[T - 1] = bestLastState;
    for (let t = T - 2; t >= 0; t--) {
        path[t] = bp[t + 1][path[t + 1]];
    }

    const tagSequence = path.map(i => states[i]);
    const taggedPairs = obs.map((x, i) => [x, tagSequence[i]]);

    return {
        observation_sequence: obs,
        best_path: tagSequence,
        tagged_pairs: taggedPairs,
        probability: bestPathProb,
        viterbi_table: V,
        backpointer_table: bp,
        states: states,
    };
}

function baumWelchAlgorithm(obsSequences, Pi, A, B, states, vocab, iterations = 5) {
    const wordToIdx = {};
    vocab.forEach((word, i) => { wordToIdx[word] = i; });

    const N = states.length;
    const M = vocab.length;

    let piNew = [...Pi];
    let aNew = A.map(row => [...row]);
    let bNew = B.map(row => [...row]);

    const likelihoodHistory = [];

    for (let iter = 0; iter < iterations; iter++) {
        // Process all sequences
        const piNumerator = Array(N).fill(0.0);
        const aNumerator = Array.from({ length: N }, () => Array(N).fill(0.0));
        const aDenominator = Array(N).fill(0.0);
        const bNumerator = Array.from({ length: N }, () => Array(M).fill(0.0));
        const bDenominator = Array(N).fill(0.0);

        for (const sequence of obsSequences) {
            const obsIdx = sequence.map(word => wordToIdx[word] ?? 0);
            const T = sequence.length;

            // Forward
            const alpha = Array.from({ length: T }, () => Array(N).fill(0.0));
            for (let s = 0; s < N; s++) {
                alpha[0][s] = piNew[s] * bNew[s][obsIdx[0]];
            }

            for (let t = 1; t < T; t++) {
                for (let s = 0; s < N; s++) {
                    for (let prevS = 0; prevS < N; prevS++) {
                        alpha[t][s] += alpha[t - 1][prevS] * aNew[prevS][s] * bNew[s][obsIdx[t]];
                    }
                }
            }

            const probObs = alpha[T - 1].reduce((a, b) => a + b, 0);
            likelihoodHistory.push(probObs);

            // Backward
            const beta = Array.from({ length: T }, () => Array(N).fill(0.0));
            for (let s = 0; s < N; s++) {
                beta[T - 1][s] = 1.0;
            }

            for (let t = T - 2; t >= 0; t--) {
                for (let s = 0; s < N; s++) {
                    for (let nextS = 0; nextS < N; nextS++) {
                        beta[t][s] += aNew[s][nextS] * bNew[nextS][obsIdx[t + 1]] * beta[t + 1][nextS];
                    }
                }
            }

            // E-step: compute gamma
            if (probObs > 0) {
                for (let t = 0; t < T; t++) {
                    for (let s = 0; s < N; s++) {
                        const gamma = (alpha[t][s] * beta[t][s]) / probObs;

                        if (t === 0) piNumerator[s] += gamma;
                        bDenominator[s] += gamma;
                        bNumerator[s][obsIdx[t]] += gamma;

                        if (t < T - 1) {
                            aDenominator[s] += gamma;
                            for (let nextS = 0; nextS < N; nextS++) {
                                const xi = (alpha[t][s] * aNew[s][nextS] * bNew[nextS][obsIdx[t + 1]] * beta[t + 1][nextS]) / probObs;
                                aNumerator[s][nextS] += xi;
                            }
                        }
                    }
                }
            }
        }

        // M-step: update parameters
        const piTotal = piNumerator.reduce((a, b) => a + b, 0);
        if (piTotal > 0) {
            piNew = piNumerator.map(x => x / piTotal);
        }

        for (let s = 0; s < N; s++) {
            if (aDenominator[s] > 0) {
                for (let nextS = 0; nextS < N; nextS++) {
                    aNew[s][nextS] = aNumerator[s][nextS] / aDenominator[s];
                }
            }
            if (bDenominator[s] > 0) {
                for (let w = 0; w < M; w++) {
                    bNew[s][w] = bNumerator[s][w] / bDenominator[s];
                }
            }
        }
    }

    return {
        updated_pi: piNew,
        updated_a: aNew,
        updated_b: bNew,
        likelihood_history: likelihoodHistory,
        iterations: iterations,
        states: states,
        vocab: vocab,
    };
}

/**
 * Solves a discrete HMM using the specified algorithm
 * @param {string} algorithm - Algorithm name: 'forward', 'backward', 'viterbi', or 'baum_welch'
 * @param {string[]} obs - Observation sequence (array of symbols)
 * @param {number[]} Pi - Initial state probabilities
 * @param {number[][]} A - State transition probabilities
 * @param {number[][]} B - Emission probabilities
 * @param {string[]} states - List of states
 * @param {string[]} vocab - List of observation symbols
 * @param {number} [iterations=5] - Number of iterations for Baum-Welch algorithm
 * @returns {object} Result of the specified algorithm
 */
function solveNumericalHMM(algorithm, obs, Pi, A, B, states, vocab, iterations = 5) {
    if (algorithm === 'forward') {
        return forwardAlgorithm(obs, Pi, A, B, states, vocab);
    } else if (algorithm === 'backward') {
        return backwardAlgorithm(obs, Pi, A, B, states, vocab);
    } else if (algorithm === 'viterbi') {
        return viterbiAlgorithm(obs, Pi, A , B, states, vocab);
    } else if (algorithm === 'baum_welch') {
        return baumWelchAlgorithm([obs], Pi, A, B, states, vocab, iterations);
    } else {
        throw new Error(`Unknown algorithm: ${algorithm}`);
    }
}

export {
    forwardAlgorithm,
    backwardAlgorithm,
    viterbiAlgorithm,
    baumWelchAlgorithm,
    solveNumericalHMM,
}