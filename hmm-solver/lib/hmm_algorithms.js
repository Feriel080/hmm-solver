import NormalDistribution from 'normal-distribution';

// Discrete Algorithms (didn't test yet)
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

function solveNumericalHMM(algorithm, obs, PiInput, AInput, BInput, states, vocab, iterations=5) {
  if (algorithm === 'forward') {
    return forwardAlgorithm(obs, PiInput, AInput, BInput, states, vocab);
  } else if (algorithm === 'backward') {
    return backwardAlgorithm(obs, PiInput, AInput, BInput, states, vocab);
  } else if (algorithm === 'viterbi') {
    return viterbiAlgorithm(obs, PiInput, AInput, BInput, states, vocab);
  } else if (algorithm === 'baum_welch') {
    return baumWelchAlgorithm([obs], PiInput, AInput, BInput, states, vocab, iterations);
  } else {
    throw new Error(`Unknown algorithm: ${algorithm}`);
  }
}

// -------------------- Continuous HMM --------------------
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

function multivariateGaussianPDF(x, mu, cov) {
  /** Multivariate Gaussian PDF for small dimensions */
  const d = x.length;
  try {
    const invCov = Math.inv(cov);
    const detCov = Math.det(cov);
    const safeDet = detCov <= 0 ? 1e-300 : detCov;
    const diff = Math.sub(x, mu);
    const temp = Math.multiply(invCov, diff);
    const exponent = -0.5 * Math.dot(diff, temp);
    const coeff = 1.0 / Math.sqrt(Math.pow(2 * Math.PI, d) * safeDet);
    return coeff * Math.exp(exponent);
  } catch (e) {
    return 1e-300;
  }
}

// 1D Algorithms
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

// N-D Algorithms
function continuousNDForward(obsValues, Pi, A, means, covariances, states) {
  /*
  obs_values   : list of list[float]  shape (T, d)
  means        : list of list[float]  shape (N, d)
  covariances  : list of list[list[float]]  shape (N, d, d)
   */
  const T = obsValues.length;
  const N = states.length;
  const B = Array.from({ length: N }, (_, s) =>
    Array.from({ length: T }, (_, t) =>
      multivariateGaussianPDF(obsValues[t], means[s], covariances[s])
    )
  );

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
    covariances: covariances,
    emission_type: 'gaussian_nd',
  };
}

function continuousNDViterbi(obsValues, Pi, A, means, covariances, states) {
  const T = obsValues.length;
  const N = states.length;
  const B = Array.from({ length: N }, (_, s) =>
    Array.from({ length: T }, (_, t) =>
      multivariateGaussianPDF(obsValues[t], means[s], covariances[s])
    )
  );

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
  const taggedPairs = obsValues.map((o, i) => [JSON.stringify(o), tagSeq[i]]);

  return {
    observation_sequence: obsValues,
    best_path: tagSeq,
    tagged_pairs: taggedPairs,
    probability: bestProb,
    viterbi_table: V,
    states: states,
    means: means,
    covariances: covariances,
    emission_type: 'gaussian_nd',
  };
}

function continuousNDBaumWelch(obsSequences, Pi, A, means, covariances, states, iterations = 5) {
  const N = states.length;
  const d = means[0].length;

  let piNew = [...Pi];
  let aNew = A.map(row => [...row]);
  let muNew = means.map(m => [...m]);
  let covNew = covariances.map(cov => cov.map(row => [...row]));
  const likelihoodHistory = [];

  for (let iter = 0; iter < iterations; iter++) {
    const piNum = Array(N).fill(0.0);
    const aNum = Array.from({ length: N }, () => Array(N).fill(0.0));
    const aDen = Array(N).fill(0.0);
    const muNum = Array.from({ length: N }, () => Array(d).fill(0.0));
    const muDen = Array(N).fill(0.0);
    const covNum = Array.from({ length: N }, () => zeros(d, d));

    for (const obsValues of obsSequences) {
      const T = obsValues.length;
      const B = Array.from({ length: N }, (_, s) =>
        Array.from({ length: T }, (_, t) =>
          multivariateGaussianPDF(obsValues[t], muNew[s], covNew[s])
        )
      );

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
          const o = obsValues[t];
          for (let s = 0; s < N; s++) {
            const g = (alpha[t][s] * beta[t][s]) / probObs;
            if (t === 0) piNum[s] += g;
            muDen[s] += g;
            muNum[s] += g * o;
            const diff = o.map((v, i) => v - muNew[s][i]);
            covNum[s] += g * Math.outer(diff, diff);
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
        muNew[s] = Math.dotDivide(muNum[s], muDen[s]);
        const devision = Math.dotDivide(covNum[s], muDen[s]);
        const identity = Math.multiply(Math.eye(d), 1e-6);
        covNew[s] = Math.add(devision, identity);
      }
    }
  }

  return {
    updated_pi: piNew,
    updated_a: aNew,
    updated_means: muNew,
    updated_covariances: covNew,
    likelihood_history: likelihoodHistory,
    iterations: iterations,
    states: states,
    emission_type: 'gaussian_nd',
  };
}

function solveContinuousHMM(algorithm, obsValues, Pi, A, means, sigmasOrCovs, states, dimension = '1d', iterations = 5) {
  if (dimension === '1d') {
    const sigmas = sigmasOrCovs;
    const obsFloat = obsValues.map(o => parseFloat(o));
    if (algorithm === 'forward') {
      return continuous1DForward(obsFloat, Pi, A, means, sigmas, states);
    } else if (algorithm === 'backward') {
      return continuous1DBackward(obsFloat, Pi, A, means, sigmas, states);
    } else if (algorithm === 'viterbi') {
      return continuous1DViterbi(obsFloat, Pi, A, means, sigmas, states);
    } else if (algorithm === 'baum_welch') {
      return continuous1DBaumWelch([obsFloat], Pi, A, means, sigmas, states, iterations);
    } else {
      throw new Error(`Unknown algorithm: ${algorithm}`);
    }
  } else {
    const covs = sigmasOrCovs;
    const obsFloat = obsValues.map(o => o.map(x => parseFloat(x)));
    if (algorithm === 'forward') {
      return continuousNDForward(obsFloat, Pi, A, means, covs, states);
    } else if (algorithm === 'viterbi') {
      return continuousNDViterbi(obsFloat, Pi, A, means, covs, states);
    } else if (algorithm === 'baum_welch') {
      return continuousNDBaumWelch([obsFloat], Pi, A, means, covs, states, iterations);
      } else {
      throw new Error(`Algorithm '${algorithm}' not supported for nd continuous HMM`);
    }
  }
}

function discretizeContinuous(obsValues, means, sigmas, states, symbols, intervals) {
  /**
   * Convert continuous Gaussian HMM to discrete by integrating N(mu, sigma^2) over symbol intervals.
   */
  const N = states.length;
  const M = symbols.length;

  const norm = new NormalDistribution(0, 1);

  const B = [];
  for (let s = 0; s < N; s++) {
    const mu = means[s];
    const sigma = Math.sqrt(sigmas[s]);
    const row = [];
    for (const [lo, hi] of intervals) {
      if (lo === -Infinity) {
        const z = (hi - mu) / sigma;
        if(z >= 0) row.push(norm.cdf(z));
        else row.push(norm.cdf(Math.abs(z)));
      } else if (hi === Infinity) {
        const z = (lo - mu) / sigma;
        if (z >= 0) row.push(1 - norm.cdf(z));
        else row.push(1 - norm.cdf(Math.abs(z)));
      } else {
        const z1 = (hi - mu) / sigma;
        const z2 = (lo - mu) / sigma;
        if (z1 >= 0) phi_hi = norm.cdf(z1);
        else phi_hi = 1 - norm.cdf(Math.abs(z1));
        if (z2 >= 0) phi_lo = norm.cdf(z2);
        else phi_lo = 1 - norm.cdf(Math.abs(z2));
        row.push(phi_hi - phi_lo);
      }
    }
    B.push(row);
  }

  const discreteObs = [];
  for (const o of obsValues) {
    const val = parseFloat(o);
    let assigned = symbols[symbols.length - 1];
    for (let m = 0; m < intervals.length; m++) {
      const [lo, hi] = intervals[m];
      if (lo <= val && val < hi) {
        assigned = symbols[m];
        break;
      }
    }
    discreteObs.push(assigned);
  }

  return {
    discrete_observation: discreteObs,
    vocab: symbols,
    b_discrete: B,
    states: states,
  };
}

// ======================== EXPORTS ========================
export {
  // Discrete
  forwardAlgorithm,
  backwardAlgorithm,
  viterbiAlgorithm,
  baumWelchAlgorithm,
  solveNumericalHMM,

  // Continuous 1D
  gaussianPDF,
  gaussianEmissionMatrix,
  continuous1DForward,
  continuous1DBackward,
  continuous1DViterbi,
  continuous1DBaumWelch,

  // Continuous N-D
  multivariateGaussianPDF,
  continuousNDForward,
  continuousNDViterbi,
  continuousNDBaumWelch,

  // Solvers & Utils
  solveContinuousHMM,
  discretizeContinuous,
};