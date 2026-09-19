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


export {
  continuousNDForward,
  continuousNDViterbi,
  continuousNDBaumWelch,
};