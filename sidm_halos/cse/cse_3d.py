import numpy as np


def rhoCSE_3d(r, s, weights=None, d=0, collapse=True):
    '''
    Computes the 3D version of the cored steep ellipse profile, or its derivatives
    '''
    if weights is not None:
        s = np.array(s)
        weights = np.array(weights)
        assert s.shape == weights.shape
        s, r = np.meshgrid(s, r)
    if d == 0:
        result = 1 / (np.pi * (r**2 + s**2)**2)
    elif d == 1:
        # first derivative
        result = - 4 * r / (np.pi * (r**2 + s**2)**3)
    elif d == 2:
        # second derivative
        result = (20 * r**2 - 4 * s**2) / (np.pi * (r**2 + s**2)**4)
    else:
        raise ValueError(f'unsupported derivative d={d}')

    if weights is not None:
        if collapse:
            return (result * weights).sum(axis=-1)
        return result * weights
    return result


def rhoCSE_m_enc(rmax, s, weights=None, collapse=True):
    '''
    Computes the 3D mass of a CSE profile out to r = rmax
    '''
    if weights is not None:
        s = np.array(s)
        weights = np.array(weights)
        assert s.shape == weights.shape
        s, rmax = np.meshgrid(s, rmax)
    result = 2 * (np.arctan(rmax/s) / s - rmax / (rmax**2 + s**2))
    if weights is not None:
        if collapse:
            return (result * weights).sum(axis=-1)
        return result * weights
    return result
