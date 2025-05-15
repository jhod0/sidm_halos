import numpy as np
from astropy import units as u
from astropy import constants
from .util import require_units


class BaryonProfile:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def density_3d(self, r):
        raise NotImplementedError

    def mass_enclosed_3d(self, r):
        raise NotImplementedError


class CallableProfile(BaryonProfile):
    def __init__(self, fn):
        # We want fn to take an astropy distance quantity and return a density
        # in units of Msun / kpc3
        # If it doesn't take the astropy ones, assume it takes floats in those
        # units, and make a wrapper to do the unit labeling
        try:
            res = fn(3*u.kpc)
            assert isinstance(res, u.Quantity)
            res.to('Msun kpc-3')
            self._func = fn
        except:
            def wrapper(r):
                r = require_units(r, 'kpc')
                return u.Unit('Msun kpc-3') * fn(r.value)
            self._func = fn

    def density_3d(self, r):
        r = require_units(r, 'kpc')
        return self._func(r)

    def mass_enclosed_3d(self, r):
        r = require_units(r, 'kpc')
        raise NotImplementedError


class DPIEProfile(BaryonProfile):
    '''
    Eliasdottir '07 dPIE profile, AKA Pseudo Jaffe profile
    '''
    def __init__(self, M, rho_0, sigma_0_e07, ra, rs):
        self._mass = require_units(M, 'Msun')
        self._ra = require_units(ra, 'kpc')
        self._rs = require_units(rs, 'kpc')
        self._rho_0 = require_units(rho_0, 'Msun kpc-3')

        self._sigma_0_e07 = require_units(sigma_0_e07, 'km / s')

    @classmethod
    def from_mass_ra_rs(cls, M, ra, rs):
        # c.f. Eliasdottir '07 eq. A11: eqn for total mass
        M = require_units(M, 'Msun')
        ra = require_units(ra, 'kpc')
        rs = require_units(rs, 'kpc')
        rho_0 = (
            M * (ra + rs)
            / (2 * np.pi**2 * ra**2 * rs**2)
        ).to('Msun kpc-3')
        sigma_0_e07 = np.sqrt((
            (4 / 3) * constants.G * np.pi * ra**2 * rs**3 * rho_0
            / ((rs - ra) * (ra + rs)**2)
        )).to('km/s')
        return cls(M=M, rho_0=rho_0, sigma_0_e07=sigma_0_e07, ra=ra, rs=rs)

    @classmethod
    def from_rho_0_ra_rs(cls, rho_0, ra, rs):
        rho_0 = require_units(rho_0, 'Msun kpc-3')
        ra = require_units(ra, 'kpc')
        rs = require_units(rs, 'kpc')
        Mtot = (2 * np.pi**2 * rho_0 * ra**2 * rs**2 / (
            ra + rs
        )).to('Msun')
        sigma_0_e07 = np.sqrt((
            (4 / 3) * constants.G * np.pi * ra**2 * rs**3 * rho_0
            / ((rs - ra) * (ra + rs)**2)
        )).to('km/s')
        return cls(M=Mtot, rho_0=rho_0, sigma_0_e07=sigma_0_e07, ra=ra, rs=rs)

    @property
    def ra(self):
        return self._ra

    @property
    def rs(self):
        return self._rs

    @property
    def mass(self):
        return self._mass

    @property
    def rho_0(self):
        return self._rho_0

    @property
    def sigma_0_e07(self):
        return self._sigma_0_e07

    @property
    def sigma_0_lenstool(self):
        # FIXME double check this
        return np.sqrt(3/2) * self._sigma_0_e07

    def density_3d(self, r):
        r = require_units(r, 'kpc')
        return self._rho_0 / ((1 + (r/self._ra)**2) * (1 + (r/self._rs)**2))

    def mass_enclosed_3d(self, r):
        r = require_units(r, 'kpc')
        return (4 * np.pi * self._rho_0 * self._ra**2 * self._rs**2 * (
            self._rs * np.arctan(r / self._rs) - self._ra * np.arctan(r / self._ra)
        ) / (self._rs**2 - self._ra**2)).to('Msun')
