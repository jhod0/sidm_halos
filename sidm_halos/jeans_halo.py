import numpy as np
from astropy import units as u
from astropy import constants
from colossus.halo import profile_nfw

from . import cosmology
from .util import require_units
from .cse.cse_decomp import (
    decompose_cse, NFWCSEDecomp, CSERepr
)
from .jeans_solvers.sidm_profiles import (
    solve_unitless_jeans, y, _interp_b_guess, _interp_c_guess
)


class OuterNFW:
    def __init__(self, M, c, z, mdef='200m'):
        self.halo = profile_nfw.NFWProfile(M=M*cosmology.h(), c=c, z=z, mdef=mdef)
        self.M = M
        self.c = c
        self.z = z
        self.mdef = mdef

    @property
    def rho_s(self):
        h = cosmology.h()
        return self.halo.par['rhos']*h**2 * u.Unit('Msun kpc-3')

    @property
    def r_s(self):
        h = cosmology.h()
        return self.halo.par['rs']/h * u.Unit('kpc')

    def __repr__(self):
        return '\n'.join([
            'Outer NFW \'skirt\' with parameters:',
            f'\tM{self.mdef:<4} = {self.M*u.Msun:.3e}',
            f'\tc{self.mdef:<4} = {self.c:.2f}',
            f'\tz     = {self.z:.2f}',
            f'\trho_s = {self.rho_s:.3e}',
            f'\tr_s   = {self.r_s:.2f}'
        ])


class InnerIsothermal:
    def __init__(self, cross_section, sigma_0, rho_0):
        self.cross_section = require_units(cross_section, 'cm2 / g')
        self.sigma_0 = require_units(sigma_0, 'km / s')
        self.rho_0 = require_units(rho_0, 'Msun kpc-3')

    def __repr__(self):
        return '\n'.join([
            'Inner isothermal region with parameters:',
            f'\tsigma/m  = {self.cross_section:.2e}',
            f'\tsigma_0  = {self.sigma_0:.1f}',
            f'\trho_0    = {self.rho_0:.3e}'
        ])


class SIDMHaloSolution:
    '''
    Represents a self-interacting dark matter halo with the semi-analytic Jeans method, using
    an outer NFW 'skirt' and inner isothermal region.
    '''
    def __init__(self, outer_nfw: OuterNFW, isothermal_region: InnerIsothermal,
                 r1, cse_decomp: CSERepr,
                 cse_xscale = None, cse_magnitude = None):
        # should have:
        #   halo params:
        #       - M200m, c200m, z
        #       - rho_NFW, rs
        #   SIDM params:
        #       - cross_section (sigma/m)
        #       - r1, sigma_0 (v. disp)
        #       - rho_0
        #   CSE decomposition of solution
        self.outer_nfw = outer_nfw
        self.isothermal_region = isothermal_region
        self.r1 = require_units(r1, 'kpc')
        self.cse_decomp = cse_decomp
        self._r = require_units(cse_xscale, 'kpc')
        self._rho = require_units(cse_magnitude, 'Msun kpc-3')

    def __repr__(self):
        return '\n'.join([
            'SIDM halo solution with:',
            f'\tr1   = {self.r1:.2f}',
            '\t' + '\n\t'.join(repr(self.isothermal_region).split('\n')),
            '\t' + '\n\t'.join(repr(self.outer_nfw).split('\n')),
        ])

    @staticmethod
    def solve_outside_in(M, c, r1, z, mdef='200m', lsq_fitter_kwargs={}, baryon_profile=None):
        '''
        Constructs a SIDM halo via the 'outside-in' method: taking a known NFW
        M200/c200 and solving what the inner isothermal part of the halo should
        look like.

        :param M: Halo mass (e.g. m200m) in Msun (no h!!)
        :param c: Halo concentration, same mdef as M
        :param r1: jeans crossover radius, kpc
        :param z: halo redshift
        '''
        halo = OuterNFW(M=M, c=c, z=z, mdef=mdef)
        r1 = require_units(r1, 'kpc')

        a = (r1 / halo.r_s).to(1).value

        if baryon_profile is None:
            guess = [_interp_b_guess(a), _interp_c_guess(a)]
            b, c = solve_unitless_jeans(a, guess=guess)
            print(a, b, c)

            def jeans_soln(xs, a, b, c):
                answer = np.empty_like(xs)
                answer[xs < a] = c*np.exp(y(xs[xs < a]/b))
                skirt_msk = xs >= a
                skirt_xs = xs[skirt_msk]
                answer[skirt_msk] = 1 / (skirt_xs * (1 + skirt_xs)**2)
                return answer

            # In units of NFW Rs
            xs = np.logspace(-6, 2, 251)
            fixed_weights = (NFWCSEDecomp._esses > 10*a)

            # Set default tolerances
            lsq_fitter_kwargs = dict(
                dict(sigma=1e-2, ftol=1e-4, xtol=1e-4),
                **lsq_fitter_kwargs
            )

            jeans_CSE_decomp, lsq_soln = decompose_cse(
                lambda x: jeans_soln(x, a, b, c), xs,
                NFWCSEDecomp._esses, NFWCSEDecomp._weights,
                fixed_weights=fixed_weights,
                return_ls_obj=True,
                # Tolerances for the least squares solver
                **lsq_fitter_kwargs
            )

            # Excellent! Solution is:
            #   - halo: has all NFW parameters
            #   - we have r1, CSE decomposition by construction
            #   - get cross section/ sigma_0, rho_0

            rho_0 = halo.rho_s*4*c
            r_0 = b * halo.r_s
            sigma_0 = np.sqrt(
                4 * np.pi * constants.G
                * rho_0 * r_0**2
            ).to('km/s')

            halo_age = cosmology.halo_age(z)
            rho_1 = 4 * halo.rho_s / (a * (1 + a)**2)
            cross_section = 1 / (
                4 * sigma_0/ np.sqrt(np.pi) * halo_age * rho_1
            )

            inner_soln = InnerIsothermal(
                cross_section=cross_section, sigma_0=sigma_0, rho_0=rho_0,
            )

            return SIDMHaloSolution(
                halo, inner_soln,
                r1, jeans_CSE_decomp,
                cse_xscale=halo.r_s, cse_magnitude=halo.rho_s*4
            )

        raise NotImplementedError

    @staticmethod
    def solve_inside_out(cross_section, N0, sigma_0, z, baryon_profile=None):
        '''
        Constructs a SIDM halo via the 'inside-out' method, solving the inner
        isothermal profile and then finding an outer NFW halo which satisfies
        the boundary conditions.

        More efficient than the outside-in method.
        '''
        if baryon_profile is None:
            pass
        raise NotImplementedError
