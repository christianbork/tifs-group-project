"""Calm vs stressed regimes from a Gaussian HMM, with forward-filtered (no look-ahead) probabilities.

hmmlearn's predict_proba / predict use the whole sequence (forward-backward,
Viterbi), so the state at t would depend on later bars. Here the state
probability at row t uses rows up to t only.
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def forward_filter(model, X: np.ndarray) -> np.ndarray:
    """Filtered state probabilities P(state_t | x_1..x_t) for one sequence, from a fitted GaussianHMM (full cov)."""
    k = model.n_components
    loglik = np.column_stack(
        [
            multivariate_normal(model.means_[j], model.covars_[j]).logpdf(X)
            for j in range(k)
        ]
    )
    log_a = np.log(model.transmat_)
    out = np.empty((len(X), k))
    alpha = np.log(model.startprob_) + loglik[0]
    out[0] = np.exp(alpha - logsumexp(alpha))
    for t in range(1, len(X)):
        alpha = logsumexp(alpha[:, None] + log_a, axis=0) + loglik[t]
        alpha -= logsumexp(alpha)  # normalise each step (log filtered probabilities)
        out[t] = np.exp(alpha)
    return out
