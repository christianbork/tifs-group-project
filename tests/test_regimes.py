import numpy as np
from hmmlearn.hmm import GaussianHMM

from tifs_group_project.regimes import forward_filter


def _fitted(seed=0):
    rng = np.random.default_rng(seed)
    X = np.r_[
        rng.normal(0, 1, (300, 2)),
        rng.normal(3, 1, (100, 2)),
        rng.normal(0, 1, (200, 2)),
    ]
    return GaussianHMM(
        n_components=2, covariance_type="full", random_state=seed, n_iter=50
    ).fit(X), X


def test_filter_uses_only_past_rows():
    model, X = _fitted()
    full, part = forward_filter(model, X), forward_filter(model, X[:350])
    assert np.allclose(
        full[:350], part
    )  # later rows do not change earlier probabilities
    assert np.allclose(full.sum(axis=1), 1)


def test_filter_matches_hmmlearn_at_last_row():
    # at the final row the smoothed and filtered probabilities coincide
    model, X = _fitted()
    assert np.allclose(
        forward_filter(model, X)[-1], model.predict_proba(X)[-1], atol=1e-6
    )
