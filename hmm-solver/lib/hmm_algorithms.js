/**
 * HMM Solver - Pure JavaScript implementation
 * Equivalent to the Python HMM algorithms (Discrete, Continuous 1D, Continuous N-D)
 * No external dependencies required.
 */

// ======================== MATH UTILITIES ========================

function gaussianPDF(o, mu, sigma) {
  /** 1D Gaussian PDF. Note: sigma here is VARIANCE (matches Python semantics). */
  const safeSigma = sigma <= 0 ? 1e-9 : sigma;
  const coeff = 1.0 / Math.sqrt(2 * Math.PI * safeSigma);
  const exponent = -0.5 * (Math.pow(o - mu, 2) / safeSigma);
  return coeff * Math.exp(exponent);
}

function erf(x) {
  /** Error function approximation (Abramowitz & Stegun) */
  const a1 =  0.254829592;
  const a2 = -0.284496736;
  const a3 =  1.421413741;
  const a4 = -1.453152027;
  const a5 =  1.061405429;
  const p  =  0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1.0 / (1.0 + p * ax);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

function normalCDF(z) {
  /** Standard normal cumulative distribution function */
  return 0.5 * (1 + erf(z / Math.sqrt(2)));
}

// -------------------- Small Matrix Linear Algebra --------------------

function zeros(rows, cols) {
  return Array.from({ length: rows }, () => Array(cols).fill(0.0));
}

function eye(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1.0 : 0.0))
  );
}

function dot(a, b) {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

function vecAdd(a, b) {
  return a.map((v, i) => v + b[i]);
}

function vecSub(a, b) {
  return a.map((v, i) => v - b[i]);
}

function vecScale(v, s) {
  return v.map(x => x * s);
}

function matVecMul(A, v) {
  return A.map(row => dot(row, v));
}

function matMul(A, B) {
  const n = A.length, m = B[0].length, p = B.length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let sum = 0;
      for (let k = 0; k < p; k++) sum += A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
  return C;
}

function matAdd(A, B) {
  return A.map((row, i) => row.map((a, j) => a + B[i][j]));
}

function matScale(A, s) {
  return A.map(row => row.map(a => a * s));
}

function transpose(A) {
  return A[0].map((_, j) => A.map(row => row[j]));
}

function outer(a, b) {
  return a.map(ai => b.map(bj => ai * bj));
}

function matDet(A) {
  /** Laplace expansion - fine for small matrices (d <= 5 typical for HMMs) */
  const n = A.length;
  if (n === 1) return A[0][0];
  if (n === 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];
  let det = 0;
  for (let j = 0; j < n; j++) {
    const minor = A.slice(1).map(row => [...row.slice(0, j), ...row.slice(j + 1)]);
    det += ((j % 2 === 0) ? 1 : -1) * A[0][j] * matDet(minor);
  }
  return det;
}

function matInv(A) {
  /** Gaussian elimination with partial pivoting */
  const n = A.length;
  const I = eye(n);
  const aug = A.map((row, i) => [...row, ...I[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxVal = Math.abs(aug[col][col]);
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > maxVal) {
        maxVal = Math.abs(aug[row][col]);
        maxRow = row;
      }
    }
    if (maxVal < 1e-12) throw new Error("Matrix is singular or nearly singular");

    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }
  return aug.map(row => row.slice(n));
}

function multivariateGaussianPDF(x, mu, cov) {
  /** Multivariate Gaussian PDF for small dimensions */
  const d = x.length;
  try {
    const invCov = matInv(cov);
    const detCov = matDet(cov);
    const safeDet = detCov <= 0 ? 1e-300 : detCov;
    const diff = vecSub(x, mu);
    const temp = matVecMul(invCov, diff);
    const exponent = -0.5 * dot(diff, temp);
    const coeff = 1.0 / Math.sqrt(Math.pow(2 * Math.PI, d) * safeDet);
    return coeff * Math.exp(exponent);
  } catch (e) {
    return 1e-300;
  }
}

function zip(a, b) {
  return a.map((x, i) => [x, b[i]]);
}

// ======================== DISCRETE HMM ========================

function forwardAlgorithm(obs, Pi, A, B, states, vocab) {
  const wordToIdx = {};
  vocab.forEach((word, i) => { wordToIdx[word] = i; });
  const obsIdx = obs.map(word => wordToIdx[word] ?? 0);

  const T = obs.length;
  const N = states.length;
  const alpha = Array.from({ length: T }, () => Array(N).fill(0.0));

  for (let s = 0; s < N; s++) {
    alpha[0][s] = Pi[s] * B[s][obsIdx[0]];
  }

  for (let t = 1; t < T; t++) {
    for (let s = 0; s < N; s++) {
      for (let prevS = 0; prevS < N; prevS++) {
        alpha[t][s] += alpha[t - 1][prevS] * A[prevS][s] * B[s][obsIdx[t]];
      }
    }
  }

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
  const beta = Array.from({ length: T }, () => Array(N).fill(0.0));

  for (let s = 0; s < N; s++) {
    beta[T - 1][s] = 1.0;
  }

  for (let t = T - 2; t >= 0; t--) {
    for (let s = 0; s < N; s++) {
      for (let nextS = 0; nextS < N; nextS++) {
        beta[t][s] += A[s][nextS] * B[nextS][obsIdx[t + 1]] * beta[t + 1][nextS];
      }
    }
  }

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
  const V = Array.from({ length: T }, () => Array(N).fill(0.0));
  const bp = Array.from({ length: T }, () => Array(N).fill(0));

  for (let s = 0; s < N; s++) {
    V[0][s] = Pi[s] * B[s][obsIdx[0]];
    bp[0][s] = 0;
  }

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

  const bestPathProb = Math.max(...V[T - 1]);
  const bestLastState = V[T - 1].indexOf(bestPathProb);

  const path = Array(T).fill(0);
  path[T - 1] = bestLastState;
  for (let t = T - 2; t >= 0; t--) {
    path[t] = bp[t + 1][path[t + 1]];
  }

  const tagSequence = path.map(i => states[i]);

  return {
    observation_sequence: obs,
    best_path: tagSequence,
    tagged_pairs: zip(obs, tagSequence),
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

      // E-step
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

    // M-step
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

function solveNumericalHMM(algorithm, obs, PiInput, AInput, BInput, states, vocab) {
  if (algorithm === 'forward') {
    return forwardAlgorithm(obs, PiInput, AInput, BInput, states, vocab);
  } else if (algorithm === 'backward') {
    return backwardAlgorithm(obs, PiInput, AInput, BInput, states, vocab);
  } else if (algorithm === 'viterbi') {
    return viterbiAlgorithm(obs, PiInput, AInput, BInput, states, vocab);
  } else if (algorithm === 'baum_welch') {
    return baumWelchAlgorithm([obs], PiInput, AInput, BInput, states, vocab);
  } else {
    throw new Error(`Unknown algorithm: ${algorithm}`);
  }
}

// ======================== CONTINUOUS 1D HMM ========================

function gaussianEmissionMatrix(obsValues, means, sigmas) {
  const N = means.length;
  const T = obsValues.length;
  return Array.from({ length: N }, (_, s) =>
    Array.from({ length: T }, (_, t) => gaussianPDF(obsValues[t], means[s], sigmas[s]))
  );
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
  return {
    observation_sequence: obsValues,
    best_path: tagSeq,
    tagged_pairs: zip(obsValues, tagSeq),
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

// ======================== CONTINUOUS N-D HMM ========================

function continuousNDForward(obsValues, Pi, A, means, covariances, states) {
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
  return {
    observation_sequence: obsValues,
    best_path: tagSeq,
    tagged_pairs: zip(obsValues.map(o => JSON.stringify(o)), tagSeq),
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
            muNum[s] = vecAdd(muNum[s], vecScale(o, g));
            const diff = vecSub(o, muNew[s]);
            covNum[s] = matAdd(covNum[s], matScale(outer(diff, diff), g));
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
        muNew[s] = vecScale(muNum[s], 1.0 / muDen[s]);
        const regularized = matAdd(matScale(covNum[s], 1.0 / muDen[s]), matScale(eye(d), 1e-6));
        covNew[s] = regularized;
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

// ======================== SOLVERS & UTILITIES ========================

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
   * NOTE: This fixes a CDF bug in the original Python (negative z-scores were handled incorrectly).
   */
  const N = states.length;
  const M = symbols.length;

  const B = [];
  for (let s = 0; s < N; s++) {
    const mu = means[s];
    const sigma = Math.sqrt(sigmas[s]);
    const row = [];
    for (const [lo, hi] of intervals) {
      if (lo === -Infinity) {
        const z = (hi - mu) / sigma;
        row.push(normalCDF(z));
      } else if (hi === Infinity) {
        const z = (lo - mu) / sigma;
        row.push(1 - normalCDF(z));
      } else {
        const z1 = (hi - mu) / sigma;
        const z2 = (lo - mu) / sigma;
        row.push(normalCDF(z1) - normalCDF(z2));
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

  // Math helpers (exported in case you need them elsewhere)
  normalCDF,
  matInv,
  matDet,
  matMul,
  matVecMul,
  vecAdd,
  vecSub,
  vecScale,
  dot,
  outer,
  eye,
  zeros,
};