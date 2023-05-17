import numpy as np


def kappaCSE_2d(r, s, weights=None, collapse=True):
    '''
    Computes the 3D version of the cored steep ellipse profile, or its derivatives
    '''
    if weights is not None:
        s = np.array(s)
        weights = np.array(weights)
        assert s.shape == weights.shape
        s, r = np.meshgrid(s, r)

    result = 1 / (2 * (r**2 + s**2)**1.5)

    if weights is not None:
        if collapse:
            return (result * weights).sum(axis=-1)
        return result * weights
    return result
