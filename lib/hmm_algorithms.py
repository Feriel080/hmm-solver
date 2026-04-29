import re
import numpy as np
from math import exp, sqrt, pi as math_pi, inf
from scipy.stats import norm

def tokenization(sentence):
    return re.findall(r'\w+', sentence.lower())

STATE_ORDER = ["DET", "NOUN", "VERB", "ADJ", "PRON"]

def HMM(corpus):
    states = set()
    vocab = set()
    for words, tags in corpus:
        states.update(tags)
        vocab.update(words)
    states = sorted(list(states), key=lambda s: STATE_ORDER.index(s) if s in STATE_ORDER else 999)
    vocab = list(vocab)
    n_states = len(states)
    n_vocab = len(vocab)

    state_idx = {tag: i for i, tag in enumerate(states)}
    word_idx = {word: i for i, word in enumerate(vocab)}

    start_count = np.zeros(n_states)
    trans_count = np.zeros((n_states, n_states))
    emit_count = np.zeros((n_states, n_vocab))

    for words, tags in corpus:
        start_count[state_idx[tags[0]]] += 1
        for i in range(len(tags)):
            s = state_idx[tags[i]]
            w = word_idx[words[i]]
            emit_count[s][w] += 1
            if i + 1 < len(tags):
                s_next = state_idx[tags[i + 1]]
                trans_count[s][s_next] += 1

    # Convert to probabilities
    Pi = start_count / len(corpus)

    A = np.zeros((n_states, n_states))
    for i in range(n_states):
        total = np.sum(trans_count[i])
        if total > 0:
            A[i] = trans_count[i] / total

    B = np.zeros((n_states, n_vocab))
    for i in range(n_states):
        total = np.sum(emit_count[i])
        if total > 0:
            B[i] = emit_count[i] / total

    return Pi.tolist(), A.tolist(), B.tolist(), states, vocab


# Discrete Algorithms
def forward_algorithm(obs, Pi, A, B, states, vocab):
    """Forward Algorithm - compute probability of observation sequence"""
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    obs_idx = [word_to_idx.get(word, 0) for word in obs]
    
    T = len(obs)
    N = len(states)
    
    # alpha[t][s] = probability
    alpha = [[0.0] * N for _ in range(T)]
    
    # Initialization
    for s in range(N):
        alpha[0][s] = Pi[s] * B[s][obs_idx[0]]
    
    # Recursion
    for t in range(1, T):
        for s in range(N):
            for prev_s in range(N):
                alpha[t][s] += alpha[t-1][prev_s] * A[prev_s][s] * B[s][obs_idx[t]]
    
    # Total probability
    prob = sum(alpha[-1])
    
    return {
        'observation_sequence': obs,
        'probability': float(prob),
        'alpha_table': alpha,
        'states': states,
    }

def backward_algorithm(obs, Pi, A, B, states, vocab):
    """Backward Algorithm - compute backward probabilities"""
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    obs_idx = [word_to_idx.get(word, 0) for word in obs]
    
    T = len(obs)
    N = len(states)
    
    # beta[t][s]
    beta = [[0.0] * N for _ in range(T)]
    
    # Initialization
    for s in range(N):
        beta[-1][s] = 1.0
    
    # Recursion (backwards)
    for t in range(T - 2, -1, -1):
        for s in range(N):
            for next_s in range(N):
                beta[t][s] += A[s][next_s] * B[next_s][obs_idx[t+1]] * beta[t+1][next_s]
    
    # Total probability
    prob = 0.0
    for s in range(N):
        prob += Pi[s] * B[s][obs_idx[0]] * beta[0][s]
    
    
    return {
        'observation_sequence': obs,
        'probability': float(prob),
        'beta_table': beta,
        'states': states,
    }

def viterbi_algorithm(obs, Pi, A, B, states, vocab):
    """Viterbi Algorithm - find most likely state sequence"""
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    obs_idx = [word_to_idx.get(word, 0) for word in obs]
    
    T = len(obs)
    N = len(states)
    
    # Viterbi variables
    V = [[0.0] * N for _ in range(T)]
    bp = [[0] * N for _ in range(T)]
    
    # Initialization
    for s in range(N):
        V[0][s] = Pi[s] * B[s][obs_idx[0]]
        bp[0][s] = 0
    
    # Recursion
    for t in range(1, T):
        for s in range(N):
            max_prob = 0.0
            max_state = 0
            for prev_s in range(N):
                prob = V[t - 1][prev_s] * A[prev_s][s] * B[s][obs_idx[t]]
                if prob > max_prob:
                    max_prob = prob
                    max_state = prev_s
            V[t][s] = max_prob
            bp[t][s] = max_state
    
    # Backtracking
    best_path_prob = max(V[-1])
    best_last_state = V[-1].index(best_path_prob)
    
    path = [0] * T
    path[-1] = best_last_state
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1][path[t + 1]]
    
    tag_sequence = [states[i] for i in path]
    
    return {
        'observation_sequence': obs,
        'best_path': tag_sequence,
        'tagged_pairs': list(zip(obs, tag_sequence)),
        'probability': float(best_path_prob),
        'viterbi_table': V,
        'backpointer_table': bp,
        'states': states,
    }

def baum_welch_algorithm(obs_sequences, Pi, A, B, states, vocab, iterations = 5):
    """Baum-Welch Algorithm - EM algorithm to re-estimate HMM parameters"""
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    N = len(states)
    M = len(vocab)
    
    Pi_new = Pi[:]
    A_new = [row[:] for row in A]
    B_new = [row[:] for row in B]
    
    likelihood_history = []
    
    for _ in range(iterations):
        # Process all sequences
        Pi_numerator = [0.0] * N
        A_numerator = [[0.0] * N for _ in range(N)]
        A_denominator = [0.0] * N
        B_numerator = [[0.0] * M for _ in range(N)]
        B_denominator = [0.0] * N
        
        for sequence in obs_sequences:
            obs_idx = [word_to_idx.get(word, 0) for word in sequence]
            T = len(sequence)
            
            # Forward 
            alpha = [[0.0] * N for _ in range(T)]
            for s in range(N):
                alpha[0][s] = Pi_new[s] * B_new[s][obs_idx[0]]
            
            for t in range(1, T):
                for s in range(N):
                    for prev_s in range(N):
                        alpha[t][s] += alpha[t-1][prev_s] * A_new[prev_s][s] * B_new[s][obs_idx[t]]
            
            prob_obs = sum(alpha[-1])
            likelihood_history.append(prob_obs)
            
            # Backward
            beta = [[0.0] * N for _ in range(T)]
            for s in range(N):
                beta[-1][s] = 1.0
            
            for t in range(T - 2, -1, -1):
                for s in range(N):
                    for next_s in range(N):
                        beta[t][s] += A_new[s][next_s] * B_new[next_s][obs_idx[t+1]] * beta[t+1][next_s]
            
            # E-step: compute gamma
            if prob_obs > 0:
                for t in range(T):
                    for s in range(N):
                        gamma = (alpha[t][s] * beta[t][s]) / prob_obs
                        
                        Pi_numerator[s] += gamma if t == 0 else 0
                        B_denominator[s] += gamma
                        B_numerator[s][obs_idx[t]] += gamma
                        
                        if t < T - 1:
                            A_denominator[s] += gamma
                            for next_s in range(N):
                                xi = (alpha[t][s] * A_new[s][next_s] * B_new[next_s][obs_idx[t+1]] * beta[t+1][next_s]) / prob_obs
                                A_numerator[s][next_s] += xi
        
        # M-step: update parameters
        Pi_total = sum(Pi_numerator)
        if Pi_total > 0:
            Pi_new = [x / Pi_total for x in Pi_numerator]
        
        for s in range(N):
            if A_denominator[s] > 0:
                for next_s in range(N):
                    A_new[s][next_s] = A_numerator[s][next_s] / A_denominator[s]
            
            if B_denominator[s] > 0:
                for w in range(M):
                    B_new[s][w] = B_numerator[s][w] / B_denominator[s]
    
    return {
        'updated_pi': Pi_new,
        'updated_a': A_new,
        'updated_b': B_new,
        'likelihood_history': likelihood_history,
        'iterations': iterations,
        'states': states,
        'vocab': vocab
    }

def solve_numerical_hmm(algorithm, obs, Pi_input, A_input, B_input, states, vocab):
    """Solve HMM with given numerical parameters"""
    
    if algorithm == 'forward':
        return forward_algorithm(obs, Pi_input, A_input, B_input, states, vocab)
    elif algorithm == 'backward':
        return backward_algorithm(obs, Pi_input, A_input, B_input, states, vocab)
    elif algorithm == 'viterbi':
        return viterbi_algorithm(obs, Pi_input, A_input, B_input, states, vocab)
    elif algorithm == 'baum_welch':
        return baum_welch_algorithm([obs], Pi_input, A_input, B_input, states, vocab)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

# Continuous HMM
def gaussian_pdf(o, mu, sigma):
    """Univariate Gaussian probability density."""
    if sigma <= 0:
        sigma = 1e-9
    return (1.0 / sqrt(2 * math_pi * sigma)) * exp(-0.5 * ((o - mu) ** 2 / sigma))
 
def gaussian_emission_matrix(obs_values, means, sigmas):
    """
    obs_values : list of float  (T observations)
    means      : list of float  (N state means)
    sigmas     : list of float  (N state sigmas)
    """
    N, T = len(means), len(obs_values)
    B = [[gaussian_pdf(obs_values[t], means[s], sigmas[s]) for t in range(T)] for s in range(N)]
    return B

def multi_dimension_gaussian_pdf(x, mu, sigma_matrix):
    """
    Multi-Dimension Gaussian pdf.
    x            : list of float  (d,)
    mu           : list of float  (d,)
    sigma_matrix : list[list[float]]  (d×d covariance)
    """
    x  = np.array(x)
    mu = np.array(mu)
    S  = np.array(sigma_matrix)
    d  = len(x)
    try:
        inv_S  = np.linalg.inv(S)
        det_S  = np.linalg.det(S)
        if det_S <= 0:
            det_S = 1e-300
        diff   = x - mu
        exponent = -0.5 * float(diff @ inv_S @ diff)
        coeff  = 1.0 / (sqrt((2 * math_pi) ** d * det_S))
        return float(coeff * exp(exponent))
    except Exception:
        return 1e-300
    
# 1D Algorithms
def continuous_1d_forward(obs_values, Pi, A, means, sigmas, states):
    T, N = len(obs_values), len(states)
    B = gaussian_emission_matrix(obs_values, means, sigmas)
    alpha = [[0.0]*N for _ in range(T)]
    for s in range(N):
        alpha[0][s] = Pi[s] * B[s][0]
    for t in range(1, T):
        for s in range(N):
            for ps in range(N):
                alpha[t][s] += alpha[t-1][ps] * A[ps][s] * B[s][t]
    prob = sum(alpha[-1])
    return {
        'observation_sequence': obs_values,
        'probability': float(prob),
        'alpha_table': alpha,
        'states': states,
        'means': means,
        'sigmas': sigmas,
        'emission_type': 'gaussian_1d',
    }

def continuous_1d_backward(obs_values, Pi, A, means, sigmas, states):
    T, N = len(obs_values), len(states)
    B = gaussian_emission_matrix(obs_values, means, sigmas)
    beta = [[0.0]*N for _ in range(T)]
    for s in range(N):
        beta[-1][s] = 1.0
    for t in range(T-2, -1, -1):
        for s in range(N):
            for ns in range(N):
                beta[t][s] += A[s][ns] * B[ns][t+1] * beta[t+1][ns]
    prob = sum(Pi[s] * B[s][0] * beta[0][s] for s in range(N))
    return {
        'observation_sequence': obs_values,
        'probability': float(prob),
        'beta_table': beta,
        'states': states,
        'means': means,
        'sigmas': sigmas,
        'emission_type': 'gaussian_1d',
    }
    
def continuous_1d_viterbi(obs_values, Pi, A, means, sigmas, states):
    T, N = len(obs_values), len(states)
    B = gaussian_emission_matrix(obs_values, means, sigmas)
    V  = [[0.0]*N for _ in range(T)]
    bp = [[0]*N   for _ in range(T)]
    for s in range(N):
        V[0][s] = Pi[s] * B[s][0]
    for t in range(1, T):
        for s in range(N):
            best_p, best_s = 0.0, 0
            for ps in range(N):
                p = V[t-1][ps] * A[ps][s] * B[s][t]
                if p > best_p:
                    best_p, best_s = p, ps
            V[t][s] = best_p
            bp[t][s] = best_s
    best_prob = max(V[-1])
    best_last = V[-1].index(best_prob)
    path = [0]*T
    path[-1] = best_last
    for t in range(T-2, -1, -1):
        path[t] = bp[t+1][path[t+1]]
    tag_seq = [states[i] for i in path]
    return {
        'observation_sequence': obs_values,
        'best_path': tag_seq,
        'tagged_pairs': list(zip(obs_values, tag_seq)),
        'probability': float(best_prob),
        'viterbi_table': V,
        'backpointer_table': bp,
        'states': states,
        'means': means,
        'sigmas': sigmas,
        'emission_type': 'gaussian_1d',
    }
    
def continuous_1d_baum_welch(obs_sequences, Pi, A, means, sigmas, states, iterations=5):
    """
    Baum-Welch for continuous 1-D Gaussian HMM.
    Re-estimates Pi, A, means, sigmas.
    """
    N = len(states)
    Pi_new    = list(Pi)
    A_new     = [row[:] for row in A]
    mu_new    = list(means)
    sigma_new = list(sigmas)
    likelihood_history = []
 
    for _ in range(iterations):
        Pi_num   = [0.0]*N
        A_num    = [[0.0]*N for _ in range(N)]
        A_den    = [0.0]*N
        mu_num   = [0.0]*N
        mu_den   = [0.0]*N
        sig_num  = [0.0]*N
 
        for obs_values in obs_sequences:
            T = len(obs_values)
            B = gaussian_emission_matrix(obs_values, mu_new, sigma_new)
 
            alpha = [[0.0]*N for _ in range(T)]
            for s in range(N):
                alpha[0][s] = Pi_new[s] * B[s][0]
            for t in range(1, T):
                for s in range(N):
                    for ps in range(N):
                        alpha[t][s] += alpha[t-1][ps] * A_new[ps][s] * B[s][t]
 
            prob_obs = sum(alpha[-1])
            likelihood_history.append(prob_obs)
 
            beta = [[0.0]*N for _ in range(T)]
            for s in range(N):
                beta[-1][s] = 1.0
            for t in range(T-2, -1, -1):
                for s in range(N):
                    for ns in range(N):
                        beta[t][s] += A_new[s][ns] * B[ns][t+1] * beta[t+1][ns]
 
            if prob_obs > 0:
                for t in range(T):
                    for s in range(N):
                        g = (alpha[t][s] * beta[t][s]) / prob_obs
                        if t == 0:
                            Pi_num[s] += g
                        mu_den[s] += g
                        mu_num[s] += g * obs_values[t]
                        sig_num[s] += g * (obs_values[t] - mu_new[s])**2
                        if t < T-1:
                            A_den[s] += g
                            for ns in range(N):
                                xi = (alpha[t][s] * A_new[s][ns] * B[ns][t+1] * beta[t+1][ns]) / prob_obs
                                A_num[s][ns] += xi
 
        Pi_total = sum(Pi_num)
        if Pi_total > 0:
            Pi_new = [x / Pi_total for x in Pi_num]
        for s in range(N):
            if A_den[s] > 0:
                for ns in range(N):
                    A_new[s][ns] = A_num[s][ns] / A_den[s]
            if mu_den[s] > 0:
                mu_new[s]    = mu_num[s]  / mu_den[s]
                sigma_new[s] = sqrt(max(sig_num[s] / mu_den[s], 1e-9))
 
    return {
        'updated_pi': Pi_new,
        'updated_a': A_new,
        'updated_means': mu_new,
        'updated_sigmas': sigma_new,
        'likelihood_history': likelihood_history,
        'iterations': iterations,
        'states': states,
        'emission_type': 'gaussian_1d',
    }
    
# N-D algorithms
def continuous_nd_forward(obs_values, Pi, A, means, covariances, states):
    """
    obs_values   : list of list[float]  shape (T, d)
    means        : list of list[float]  shape (N, d)
    covariances  : list of list[list[float]]  shape (N, d, d)
    """
    T, N = len(obs_values), len(states)
    B = [[multi_dimension_gaussian_pdf(obs_values[t], means[s], covariances[s])
          for t in range(T)] for s in range(N)]
    alpha = [[0.0]*N for _ in range(T)]
    for s in range(N):
        alpha[0][s] = Pi[s] * B[s][0]
    for t in range(1, T):
        for s in range(N):
            for ps in range(N):
                alpha[t][s] += alpha[t-1][ps] * A[ps][s] * B[s][t]
    prob = sum(alpha[-1])
    return {
        'observation_sequence': obs_values,
        'probability': float(prob),
        'alpha_table': alpha,
        'states': states,
        'means': means,
        'covariances': covariances,
        'emission_type': 'gaussian_nd',
    }
    
def continuous_nd_viterbi(obs_values, Pi, A, means, covariances, states):
    T, N = len(obs_values), len(states)
    B = [[multi_dimension_gaussian_pdf(obs_values[t], means[s], covariances[s])
          for t in range(T)] for s in range(N)]
    V  = [[0.0]*N for _ in range(T)]
    bp = [[0]*N   for _ in range(T)]
    for s in range(N):
        V[0][s] = Pi[s] * B[s][0]
    for t in range(1, T):
        for s in range(N):
            best_p, best_s = 0.0, 0
            for ps in range(N):
                p = V[t-1][ps] * A[ps][s] * B[s][t]
                if p > best_p:
                    best_p, best_s = p, ps
            V[t][s] = best_p
            bp[t][s] = best_s
    best_prob = max(V[-1])
    best_last = V[-1].index(best_prob)
    path = [0]*T
    path[-1] = best_last
    for t in range(T-2, -1, -1):
        path[t] = bp[t+1][path[t+1]]
    tag_seq = [states[i] for i in path]
    return {
        'observation_sequence': obs_values,
        'best_path': tag_seq,
        'tagged_pairs': list(zip([str(o) for o in obs_values], tag_seq)),
        'probability': float(best_prob),
        'viterbi_table': V,
        'states': states,
        'means': means,
        'covariances': covariances,
        'emission_type': 'gaussian_nd',
    }

def continuous_nd_baum_welch(obs_sequences, Pi, A, means, covariances, states, iterations=5):
    N = len(states)
    d = len(means[0])
    Pi_new  = list(Pi)
    A_new   = [row[:] for row in A]
    mu_new  = [list(m) for m in means]
    cov_new = [[list(row) for row in cov] for cov in covariances]
    likelihood_history = []
 
    for _ in range(iterations):
        Pi_num  = [0.0]*N
        A_num   = [[0.0]*N for _ in range(N)]
        A_den   = [0.0]*N
        mu_num  = [np.zeros(d) for _ in range(N)]
        mu_den  = [0.0]*N
        cov_num = [np.zeros((d, d)) for _ in range(N)]
 
        for obs_values in obs_sequences:
            T = len(obs_values)
            B = [[multi_dimension_gaussian_pdf(obs_values[t], mu_new[s], cov_new[s])
                  for t in range(T)] for s in range(N)]
 
            alpha = [[0.0]*N for _ in range(T)]
            for s in range(N):
                alpha[0][s] = Pi_new[s] * B[s][0]
            for t in range(1, T):
                for s in range(N):
                    for ps in range(N):
                        alpha[t][s] += alpha[t-1][ps] * A_new[ps][s] * B[s][t]
 
            prob_obs = sum(alpha[-1])
            likelihood_history.append(float(prob_obs))
 
            beta = [[0.0]*N for _ in range(T)]
            for s in range(N):
                beta[-1][s] = 1.0
            for t in range(T-2, -1, -1):
                for s in range(N):
                    for ns in range(N):
                        beta[t][s] += A_new[s][ns] * B[ns][t+1] * beta[t+1][ns]
 
            if prob_obs > 0:
                for t in range(T):
                    o = np.array(obs_values[t])
                    for s in range(N):
                        g = (alpha[t][s] * beta[t][s]) / prob_obs
                        if t == 0:
                            Pi_num[s] += g
                        mu_den[s] += g
                        mu_num[s] += g * o
                        diff = o - np.array(mu_new[s])
                        cov_num[s] += g * np.outer(diff, diff)
                        if t < T-1:
                            A_den[s] += g
                            for ns in range(N):
                                xi = (alpha[t][s] * A_new[s][ns] * B[ns][t+1] * beta[t+1][ns]) / prob_obs
                                A_num[s][ns] += xi
 
        Pi_total = sum(Pi_num)
        if Pi_total > 0:
            Pi_new = [x / Pi_total for x in Pi_num]
        for s in range(N):
            if A_den[s] > 0:
                for ns in range(N):
                    A_new[s][ns] = A_num[s][ns] / A_den[s]
            if mu_den[s] > 0:
                mu_new[s]  = (mu_num[s] / mu_den[s]).tolist()
                cov_new[s] = (cov_num[s] / mu_den[s] + np.eye(d) * 1e-6).tolist()
 
    return {
        'updated_pi': Pi_new,
        'updated_a': A_new,
        'updated_means': mu_new,
        'updated_covariances': cov_new,
        'likelihood_history': likelihood_history,
        'iterations': iterations,
        'states': states,
        'emission_type': 'gaussian_nd',
    }
    
def solve_continuous_hmm(algorithm, obs_values, Pi, A, means, sigmas_or_covs, states, dimension='1d', iterations=5):
    """
    dimension : '1d' or 'nd'
    sigmas_or_covs:
        1d  -> list of float (one sigma per state)
        nd  -> list of list[list[float]] (covariance matrix per state)
    """
    if dimension == '1d':
        sigmas = sigmas_or_covs
        obs_float = [float(o) for o in obs_values]
        if algorithm == 'forward':
            return continuous_1d_forward(obs_float, Pi, A, means, sigmas, states)
        elif algorithm == 'backward':
            return continuous_1d_backward(obs_float, Pi, A, means, sigmas, states)
        elif algorithm == 'viterbi':
            return continuous_1d_viterbi(obs_float, Pi, A, means, sigmas, states)
        elif algorithm == 'baum_welch':
            return continuous_1d_baum_welch([obs_float], Pi, A, means, sigmas, states, iterations)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    else:  # nd
        covs = sigmas_or_covs
        obs_float = [[float(x) for x in o] for o in obs_values]
        if algorithm == 'forward':
            return continuous_nd_forward(obs_float, Pi, A, means, covs, states)
        elif algorithm == 'viterbi':
            return continuous_nd_viterbi(obs_float, Pi, A, means, covs, states)
        elif algorithm == 'baum_welch':
            return continuous_nd_baum_welch([obs_float], Pi, A, means, covs, states, iterations)
        else:
            raise ValueError(f"Algorithm '{algorithm}' not supported for nd continuous HMM")
 
def discretize_continuous(obs_values, means, sigmas, states, symbols, intervals):
    """
    Convert continuous Gaussian HMM to discrete by integrating N(μ,σ²) over symbol intervals.
    
    symbols   : list of symbol names e.g. ['Cold', 'Mild', 'Hot']
    intervals : list of (lower, upper) bounds per symbol, use -inf/inf for edges
                e.g. [(-inf, 20), (20, 24), (24, inf)]
    means     : list of float, one per state
    sigmas    : list of float, one per state
    """
    N = len(states)
    M = len(symbols)
    
    B = []
    for s in range(N):
        mu, sigma = means[s], sigmas[s]
        row = []
        for lo, hi in intervals:
            phi_hi = norm.cdf((hi  - mu) / sigma) if hi  != inf  else 1.0
            phi_lo = norm.cdf((lo  - mu) / sigma) if lo  != -inf else 0.0
            row.append(float(phi_hi - phi_lo))
            
        # normalize row (floating point safety)
        # total = sum(row)
        # row = [v / total if total > 0 else 1/M for v in row]
        B.append(row)
        
        # Discretize the observation sequence
    discrete_obs = []
    for o in obs_values:
        o = float(o)
        assigned = symbols[-1]  # default to last
        for m, (lo, hi) in enumerate(intervals):
            if lo <= o < hi:
                assigned = symbols[m]
                break
        discrete_obs.append(assigned)

    return {
        'discrete_observation': discrete_obs,
        'vocab': symbols,
        'b_discrete': B,
        'states': states,
    }

# Demo corpus
DEMO_CORPUS = [
    (["la", "petite", "brise", "la", "glace"], ["DET", "NOUN", "VERB", "DET", "NOUN"]),
    (["la", "petite", "brise", "la", "glace"], ["DET", "NOUN", "VERB", "DET", "NOUN"]),
    (["la", "petite", "brise", "la", "glace"], ["DET", "ADJ", "NOUN", "PRON", "VERB"]), 
    (["la", "petite", "glace"], ["DET", "ADJ", "NOUN"]),
    (["la", "petite", "brise"], ["DET", "ADJ", "NOUN"]),
]

def solve_demo(sentence = "la petite brise la glace"):
    """Solve the demo ambiguous phrase"""
    tokens = tokenization(sentence)
    
    Pi, A, B, states, vocab = HMM(DEMO_CORPUS)
    oov = [t for t in tokens if t not in vocab]
    result = viterbi_algorithm(tokens, Pi, A, B, states, vocab)
    result['corpus'] = {
        'states': states,
        'vocab': vocab,
        'pi': Pi,
        'a': A,
        'b': B
    }
    result['baum_welch'] = baum_welch_algorithm([tokens], Pi, A, B, states, vocab, iterations=3)
    result['oov_words'] = oov 
    
    return result
