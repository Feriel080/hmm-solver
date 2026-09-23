function gaussianPDF(o, mu, sigma) {
  const safeSigma = sigma <= 0 ? 1e-9 : sigma;
  const coeff = 1.0 / Math.sqrt(2 * Math.PI * safeSigma);
  const exponent = -0.5 * (Math.pow(o - mu, 2) / safeSigma);
  return coeff * Math.exp(exponent);
}

function gaussianEmissionMatrix(obsValues, means, sigmas) {
  /************************************
   * obsValues: T ovservations
   * means: N state means
   * sigmas: N state variances
   */

  const N = means.length;
  const T = obsValues.length;
  const B = Array.from({ length: N }, (_, s) =>
    Array.from({ length: T }, (_, t) => gaussianPDF(obsValues[t], means[s], sigmas[s]))
  );
  return B;
}

function continuous1DForward(obsValues, Pi, A, means, sigmas, states) {
    const T = obsValues.length;
    const N = states.length;
    const B = gaussianEmissionMatrix(obsValues, means, sigmas);
    const alpha = Array.from({ length: T }, () => Array(N).fill(0.0));

    for (let s = 0; s < N; s++) {
        alpha[0][s] = Pi[s] * B[s][0];
    }

    for (let t = 1; t < T; t++) {
        for (let s = 0; s < N; s++) {
            for (let ps = 0; ps < N; ps++) {
                alpha[t][s] += alpha[t - 1][ps] * A[ps][s] * B[s][t];
            }
        }
    }

    const prob = alpha[T - 1].reduce((a, b) => a + b, 0);
    return {
        observation_sequence: obsValues,
        probability: prob,
        alpha_table: alpha,
        states: states,
        means: means,
        sigmas: sigmas,
        emission_type: 'gaussian_1d',
    };
}

function continuous1DBackward(obsValues, Pi, A, means, sigmas, states) {
    const T = obsValues.length;
    const N = states.length;
    const B = gaussianEmissionMatrix(obsValues, means, sigmas);
    const beta = Array.from({ length: T }, () => Array(N).fill(0.0));

    for (let s = 0; s < N; s++) {
        beta[T - 1][s] = 1.0;
    }

    for (let t = T - 2; t >= 0; t--) {
        for (let s = 0; s < N; s++) {
            for (let ns = 0; ns < N; ns++) {
                beta[t][s] += A[s][ns] * B[ns][t + 1] * beta[t + 1][ns];
            }
        }
    }

    const prob = Array.from({ length: N }, (_, s) => Pi[s] * B[s][0] * beta[0][s]).reduce((a, b) => a + b, 0);
    return {
        observation_sequence: obsValues,
        probability: prob,
        beta_table: beta,
        states: states,
        means: means,
        sigmas: sigmas,
        emission_type: 'gaussian_1d',
    };
}

function continuous1DViterbi(obsValues, Pi, A, means, sigmas, states) {
    const T = obsValues.length;
    const N = states.length;
    const B = gaussianEmissionMatrix(obsValues, means, sigmas);
    const V = Array.from({ length: T }, () => Array(N).fill(0.0));
    const bp = Array.from({ length: T }, () => Array(N).fill(0));

    for (let s = 0; s < N; s++) {
        V[0][s] = Pi[s] * B[s][0];
    }

    for (let t = 1; t < T; t++) {
        for (let s = 0; s < N; s++) {
            let bestP = 0.0, bestS = 0;
            for (let ps = 0; ps < N; ps++) {
                const p = V[t - 1][ps] * A[ps][s] * B[s][t];
                if (p > bestP) {
                    bestP = p;
                    bestS = ps;
                }
            }
            V[t][s] = bestP;
            bp[t][s] = bestS;
        }
    }

    const bestProb = Math.max(...V[T - 1]);
    const bestLast = V[T - 1].indexOf(bestProb);
    const path = Array(T).fill(0);
    path[T - 1] = bestLast;

    for (let t = T - 2; t >= 0; t--) {
        path[t] = bp[t + 1][path[t + 1]];
    }

    const tagSeq = path.map(i => states[i]);
    const taggedPairs = obsValues.map((o, i) => [o, tagSeq[i]]);
    return {
        observation_sequence: obsValues,
        best_path: tagSeq,
        tagged_pairs: taggedPairs,
        probability: bestProb,
        viterbi_table: V,
        backpointer_table: bp,
        states: states,
        means: means,
        sigmas: sigmas,
        emission_type: 'gaussian_1d',
    };
}

function continuous1DBaumWelch(obsSequences, Pi, A, means, sigmas, states, iterations = 5) {
    const N = states.length;
    let piNew = [...Pi];
    let aNew = A.map(row => [...row]);
    let muNew = [...means];
    let sigmaNew = [...sigmas];
    const likelihoodHistory = [];

    for (let iter = 0; iter < iterations; iter++) {
        const piNum = Array(N).fill(0.0);
        const aNum = Array.from({ length: N }, () => Array(N).fill(0.0));
        const aDen = Array(N).fill(0.0);
        const muNum = Array(N).fill(0.0);
        const muDen = Array(N).fill(0.0);
        const sigNum = Array(N).fill(0.0);

        for (const obsValues of obsSequences) {
            const T = obsValues.length;
            const B = gaussianEmissionMatrix(obsValues, muNew, sigmaNew);

            // Forward
            const alpha = Array.from({ length: T }, () => Array(N).fill(0.0));
            for (let s = 0; s < N; s++) {
                alpha[0][s] = piNew[s] * B[s][0];
            }
            for (let t = 1; t < T; t++) {
                for (let s = 0; s < N; s++) {
                    for (let ps = 0; ps < N; ps++) {
                        alpha[t][s] += alpha[t - 1][ps] * aNew[ps][s] * B[s][t];
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
                    for (let ns = 0; ns < N; ns++) {
                        beta[t][s] += aNew[s][ns] * B[ns][t + 1] * beta[t + 1][ns];
                    }
                }
            }

            if (probObs > 0) {
                for (let t = 0; t < T; t++) {
                    for (let s = 0; s < N; s++) {
                        const g = (alpha[t][s] * beta[t][s]) / probObs;
                        if (t === 0) piNum[s] += g;
                        muDen[s] += g;
                        muNum[s] += g * obsValues[t];
                        sigNum[s] += g * Math.pow(obsValues[t] - muNew[s], 2);
                        if (t < T - 1) {
                            aDen[s] += g;
                            for (let ns = 0; ns < N; ns++) {
                                const xi = (alpha[t][s] * aNew[s][ns] * B[ns][t + 1] * beta[t + 1][ns]) / probObs;
                                aNum[s][ns] += xi;
                            }
                        }
                    }
                }
            }
        }

        const piTotal = piNum.reduce((a, b) => a + b, 0);
        if (piTotal > 0) {
            piNew = piNum.map(x => x / piTotal);
        }

        for (let s = 0; s < N; s++) {
            if (aDen[s] > 0) {
                for (let ns = 0; ns < N; ns++) {
                    aNew[s][ns] = aNum[s][ns] / aDen[s];
                }
            }

            if (muDen[s] > 0) {
                muNew[s] = muNum[s] / muDen[s];
                sigmaNew[s] = Math.sqrt(Math.max(sigNum[s] / muDen[s], 1e-9));
            }
        }
    }

    return {
        updated_pi: piNew,
        updated_a: aNew,
        updated_means: muNew,
        updated_sigmas: sigmaNew,
        likelihood_history: likelihoodHistory,
        iterations: iterations,
        states: states,
        emission_type: 'gaussian_1d',
    };
}

export {
    continuous1DForward,
    continuous1DBackward,
    continuous1DViterbi,
    continuous1DBaumWelch,
}