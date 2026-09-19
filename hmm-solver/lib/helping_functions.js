import NormalDistribution from 'normal-distribution';

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
        if (z >= 0) row.push(norm.cdf(z));
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

export {
  gaussianEmissionMatrix,
  multivariateGaussianPDF,
  solveContinuousHMM,
  discretizeContinuous,
}