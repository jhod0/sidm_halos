import numpy as np
from astropy import units as u
from astropy import constants
from colossus.halo import profile_nfw

from . import cosmology
from .util import require_units
from .cse.cse_decomp import (
    decompose_cse, decompose_analytic_jeans, CSERepr
)
from .jeans_solvers.sidm_profiles import (
    solve_unitless_jeans, y, _interp_b_guess, _interp_c_guess
)
from .jeans_solvers import sidm_profiles as _sidm_solved


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
    def __init__(self, cross_section, sigma_0, rho_0, halo_age):
        self.cross_section = require_units(cross_section, 'cm2 / g')
        self.sigma_0 = require_units(sigma_0, 'km / s')
        self.rho_0 = require_units(rho_0, 'Msun kpc-3')
        self.halo_age = require_units(halo_age, 'Gyr')

    @property
    def N0(self):
        return (
            4 * self.sigma_0 / np.sqrt(np.pi)
            * self.rho_0 * self.cross_section * self.halo_age
        ).to(1)

    def __repr__(self):
        return '\n'.join([
            'Inner isothermal region with parameters:',
            f'\tsigma/m  = {self.cross_section:.2e}',
            f'\tsigma_0  = {self.sigma_0:.1f}',
            f'\trho_0    = {self.rho_0:.3e}',
            f'\tN0       = {self.N0:.2f}',
            f'\tage      = {self.halo_age:.2f}',
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
        halo_age = cosmology.halo_age(z)
        r1 = require_units(r1, 'kpc')

        a = (r1 / halo.r_s).to(1).value

        if baryon_profile is None:
            guess = [_interp_b_guess(a), _interp_c_guess(a)]
            b, c = solve_unitless_jeans(a, guess=guess)
            # print(a, b, c)

            # Set default tolerances
            lsq_fitter_kwargs = dict(
                dict(sigma=1e-2, ftol=1e-4, xtol=1e-4),
                **lsq_fitter_kwargs
            )

            jeans_CSE_decomp, lsq_soln = decompose_analytic_jeans(
                a, b, c,
                return_ls_obj=True,
                # Tolerances for the least squares solver
                **lsq_fitter_kwargs,
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

            rho_1 = 4 * halo.rho_s / (a * (1 + a)**2)
            cross_section = 1 / (
                4 * sigma_0/ np.sqrt(np.pi) * halo_age * rho_1
            )

            inner_soln = InnerIsothermal(
                cross_section=cross_section, sigma_0=sigma_0, rho_0=rho_0, halo_age=halo_age
            )

            return SIDMHaloSolution(
                halo, inner_soln,
                r1, jeans_CSE_decomp,
                cse_xscale=halo.r_s, cse_magnitude=halo.rho_s*4
            )

        raise NotImplementedError

    @staticmethod
    def solve_inside_out(cross_section, N0, sigma_0, z, mdef='200m', lsq_fitter_kwargs={}, baryon_profile=None):
        '''
        Constructs a SIDM halo via the 'inside-out' method, solving the inner
        isothermal profile and then finding an outer NFW halo which satisfies
        the boundary conditions.

        More efficient than the outside-in method.
        '''
        h = cosmology.h()
        halo_age = cosmology.halo_age(z)
        cross_section = require_units(cross_section, 'cm2/g')
        sigma_0 = require_units(sigma_0, 'km/s')

        if baryon_profile is None:
            rho_0 = (
                N0 / (4 * sigma_0 / np.sqrt(np.pi) * cross_section * halo_age)
            ).to('Msun kpc-3')

            # FIXME get more exact x1
            # x1 = r1 / r0
            idx_soln = np.searchsorted(
                # Sign funniness for searchsorted to do the right thing
                -_sidm_solved.solution[:, 0],
                np.log(N0)
            )
            x1 = _sidm_solved.xspan[idx_soln]

            r_0 = require_units(
                sigma_0 / np.sqrt(4 * np.pi * constants.G * rho_0),
                'kpc'
            )

            # Crossover radius and density thereat
            r_1 = x1 * r_0
            rho_1 = rho_0 / N0

            inner_soln = InnerIsothermal(
                cross_section, sigma_0, rho_0, halo_age
            )

            # Great - now solve for outer NFW
            # We have unitful rho_1, r_1
            boundary_density = np.exp(y(x1))
            mass_enclosed = _sidm_solved.mass_interp(x1)

            # We know x1 = r1/r0 = b/a
            # and we have a relation for b/a vs. a
            # So solve for a
            a = _sidm_solved._interp_a_from_a_over_b(x1)
            b = a / x1
            c = _sidm_solved._interp_c_guess(a)
            # print(a, b, c)

            # NFW params
            rho_s = rho_0 / c / 4
            r_s = r_1 / a

            # print(rho_s)
            # print(r_s)

            halo = profile_nfw.NFWProfile(
                rhos=(rho_s/h**2).to('Msun kpc-3').value, rs=(r_s*h).to('kpc').value, z=z
            )
            Rhalo, Mhalo = halo.RMDelta(z, mdef)
            Rhalo /= h
            Mhalo /= h
            # print(Rhalo, Mhalo)

            chalo = (Rhalo*u.kpc / r_s).to(1).value
            # print(chalo)

            outer_nfw = OuterNFW(Mhalo, chalo, z, mdef=mdef)

            # Set default tolerances
            lsq_fitter_kwargs = dict(
                dict(sigma=1e-2, ftol=1e-4, xtol=1e-4),
                **lsq_fitter_kwargs
            )

            # TODO
            a = (r_1 / r_s).to(1).value
            b = (r_0 / r_s).to(1).value
            c = (rho_0 / (4 * rho_s)).to(1).value
            jeans_CSE_decomp, lsq_soln = decompose_analytic_jeans(
                a, b, c,
                return_ls_obj=True,
                **lsq_fitter_kwargs
            )

            return SIDMHaloSolution(
                outer_nfw, inner_soln,
                r_1, jeans_CSE_decomp,
                cse_xscale=r_s, cse_magnitude=rho_s*4
            )

        raise NotImplementedError
