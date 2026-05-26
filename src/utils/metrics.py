import numpy as np


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def cmapss_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = y_pred - y_true

    score = 0.0
    for e in errors:
        if e < 0:
            score += np.exp(-e / 13.0) - 1.0
        else:
            score += np.exp(e / 10.0) - 1.0
    return score