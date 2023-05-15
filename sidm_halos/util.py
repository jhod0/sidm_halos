from astropy import units as u
from astropy import constants
import numpy as np


def require_units(obj, units):
    if isinstance(obj, u.Quantity):
        return obj.to(units)
    elif isinstance(obj, (float, int, np.ndarray)):
        return obj * u.Unit(units)
    else:
        raise ValueError(f'not sure how to give units to {obj}')
