import numpy as np
from astropy import units as u
from astropy import constants
from colossus.halo import profile_nfw
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from scipy import optimize as opt

from . import cosmology
from .util import require_units
from .cse.cse_decomp import (
    decompose_integrated_jeans, decompose_analytic_jeans, CSERepr
)
from .jeans_solvers import sidm_profiles as _sidm_solved
from .jeans_solvers import solver_with_baryons as _baryons_solver


class OuterNFW:
    def __init__(self, M, c, z, mdef='200m'):
        self.M = require_units(M, 'Msun')
        self.halo = profile_nfw.NFWProfile(M=self.M.value*cosmology.h(), c=c, z=z, mdef=mdef)
        self.c = c
        self.z = z
        self.mdef = mdef

    @staticmethod
    def solve_from_boundary(r1, rho_1, Menc, z, mdef='200m'):
        '''
        Returns the NFW halo that satisfies:

            - rho(r1) = rho_1
            - M(<r1) = Menc
        '''
        r1 = require_units(r1, 'kpc')
        rho_1 = require_units(rho_1, 'Msun kpc-3')
        Menc = require_units(Menc, 'Msun')

        K = (Menc / (rho_1 * r1**3)).to(1).value
        def f(lna):
            a = np.exp(lna)
            # Should equal K
            ratio = 4 * np.pi / a**3 * (np.log(1 + a) - a / (1 + a)) * a * (1 + a)**2
            return 100 * np.log(ratio / K)

        solved_a = opt.root(f, [0])
        a = np.exp(solved_a.x).reshape(())

        r_s = r1 / a
        rho_s = rho_1 * a * (1 + a)**2

        h = cosmology.h()
        halo = profile_nfw.NFWProfile(
            rhos=(rho_s/h**2).to('Msun kpc-3').value, rs=(r_s*h).to('kpc').value, z=z
        )
        Rhalo, Mhalo = halo.RMDelta(z, mdef)
        Rhalo /= h
        Mhalo /= h
        chalo = (Rhalo*u.kpc / r_s).to(1).value

        return OuterNFW(M=Mhalo, c=chalo, z=z, mdef=mdef)

    def density_3d(self, r):
        '''
        NFW 3D density at a radius r, in Msun / kpc3
        '''
        r = require_units(r, 'kpc')
        x = (r / self.r_s).to(1).value
        return self.rho_s / (x * (1 + x)**2)

    def mass_enclosed_3d(self, r):
        '''
        3D mass enclosed within a radius r
        '''
        r = require_units(r, 'kpc')
        x = (r / self.r_s).to(1).value
        return (4 * np.pi * self.rho_s * self.r_s**3 * (
            np.log(1 + x) - x / (1 + x)
        )).to('Msun')

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
            f'\tM{self.mdef:<4} = {self.M:.3e}',
            f'\tc{self.mdef:<4} = {self.c:.2f}',
            f'\tz     = {self.z:.2f}',
            f'\trho_s = {self.rho_s:.3e}',
            f'\tr_s   = {self.r_s:.2f}'
        ])


class InnerIsothermal:
    def __init__(self, cross_section, sigma_0, rho_0, halo_age,
                 solution_interp=None, rscale_interp=None, mag_interp=None):
        self.cross_section = require_units(cross_section, 'cm2 / g')
        self.sigma_0 = require_units(sigma_0, 'km / s')
        self.rho_0 = require_units(rho_0, 'Msun kpc-3')
        self.halo_age = require_units(halo_age, 'Gyr')
        if solution_interp is not None:
            assert rscale_interp is not None
            assert mag_interp is not None

            self.rscale_interp = require_units(rscale_interp, 'kpc')
            self.mag_interp = require_units(mag_interp, 'Msun kpc-3')
            self.solution_interp = solution_interp
        else:
            self.solution_interp = None

    def density_3d(self, r):
        r = require_units(r, 'kpc')
        x = (r / self.rscale_interp).to(1).value
        return self.mag_interp * np.exp(self.solution_interp(x))

    @property
    def N0(self):
        return (
            4 * self.sigma_0 / np.sqrt(np.pi)
            * self.rho_0 * self.cross_section * self.halo_age
        ).to(1)

    def __repr__(self):
        has_interp = self.solution_interp is not None
        return '\n'.join([
            'Inner isothermal region with parameters:',
            f'\tsigma/m  = {self.cross_section:.2e}',
            f'\tsigma_0  = {self.sigma_0:.1f}',
            f'\trho_0    = {self.rho_0:.3e}',
            f'\tN0       = {self.N0:.2f}',
            f'\tage      = {self.halo_age:.2f}',
            f'\tinterp   = {has_interp}',
        ])


class SIDMHaloSolution:
    '''
    Represents a self-interacting dark matter halo with the semi-analytic Jeans method, using
    an outer NFW 'skirt' and inner isothermal region.
    '''
    def __init__(self, outer_nfw: OuterNFW, isothermal_region: InnerIsothermal,
                 r1, cse_decomp: CSERepr):
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

    def __repr__(self):
        return '\n'.join([
            'SIDM halo solution with:',
            f'\tr1   = {self.r1:.2f}',
            '\t' + '\n\t'.join(repr(self.isothermal_region).split('\n')),
            '\t' + '\n\t'.join(repr(self.outer_nfw).split('\n')),
        ])

    @staticmethod
    def solve_outside_in(M, c, r1, z, mdef='200m', baryon_profile=None,
                         lsq_fitter_kwargs={}, solver='bvp', solver_kwargs={}):
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

        # Set default tolerances
        lsq_fitter_kwargs = dict(
            dict(sigma=1e-2, ftol=1e-4, xtol=1e-4),
            **lsq_fitter_kwargs
        )

        a = (r1 / halo.r_s).to(1).value

        if baryon_profile is None:
            guess = _sidm_solved.guess_b_c(a)
            b, c = _sidm_solved.solve_unitless_jeans(a, guess=guess)

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

            rho_0 = halo.rho_s*c
            r_0 = b * halo.r_s
            sigma_0 = np.sqrt(
                4 * np.pi * constants.G
                * rho_0 * r_0**2
            ).to('km/s')

            rho_1 = halo.rho_s / (a * (1 + a)**2)
            cross_section = 1 / (
                4 * sigma_0/ np.sqrt(np.pi) * halo_age * rho_1
            )

            inner_soln = InnerIsothermal(
                cross_section=cross_section, sigma_0=sigma_0, rho_0=rho_0,
                halo_age=halo_age,
                solution_interp=_sidm_solved.y,
                rscale_interp=r_0,
                mag_interp=rho_0,
            )

            return SIDMHaloSolution(
                halo, inner_soln,
                r1, jeans_CSE_decomp,
            )

        # Now the case with baryons
        Menc = halo.mass_enclosed_3d(r1)
        rho_1 = halo.density_3d(r1)

        if solver == 'ivp':
            # TODO allow tuning inner radius etc
            result_integrand, result_params, result_root = _baryons_solver.solve_outside_in(
                r1, Menc, rho_1, halo_age, baryon_profile,
                **solver_kwargs
            )
        elif solver == 'bvp':
            result_integrand, result_params = _baryons_solver.solve_outside_in_as_bvp(
                r1, Menc, rho_1, halo_age, baryon_profile,
                **solver_kwargs
            )
        else:
            raise ValueError(f'unkown outside-in solver {solver}')

        inner_soln = InnerIsothermal(
            cross_section=result_params['cross_section'], sigma_0=result_params['sigma_0'],
            rho_0=result_params['rho_0'], halo_age=halo_age,
            solution_interp=lambda x: result_integrand.sol(x)[0],
            rscale_interp=result_params['r0'],
            mag_interp=result_params['rho_0'],
        )

        a = (r1 / halo.r_s).to(1).value
        b = (result_params['r0'] / halo.r_s).to(1).value
        c = (result_params['rho_0'] / halo.rho_s).to(1).value

        jeans_CSE_decomp = decompose_integrated_jeans(
            lambda x: result_integrand.sol(x)[0], a, b, c,
            **lsq_fitter_kwargs
        )

        return SIDMHaloSolution(
            halo, inner_soln, r1, jeans_CSE_decomp,
        )

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

        # Set default tolerances
        lsq_fitter_kwargs = dict(
            dict(sigma=1e-2, ftol=1e-4, xtol=1e-4),
            **lsq_fitter_kwargs
        )

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
                cross_section, sigma_0, rho_0, halo_age,
                solution_interp=_sidm_solved.y,
                rscale_interp=r_0,
                mag_interp=rho_0,
            )

            # Great - now solve for outer NFW
            # We have unitful rho_1, r_1
            boundary_density = np.exp(_sidm_solved.y(x1)) * rho_0
            mass_enclosed = 4 * np.pi * _sidm_solved.mass_interp(x1) * (rho_0 * r_0**3)

            outer_nfw = OuterNFW.solve_from_boundary(
                r_1, boundary_density, mass_enclosed, z=z, mdef=mdef
            )
            r_s = outer_nfw.r_s
            rho_s = outer_nfw.rho_s

            a = (r_1 / r_s).to(1).value
            b = (r_0 / r_s).to(1).value
            c = (rho_0 / (rho_s)).to(1).value
            jeans_CSE_decomp, lsq_soln = decompose_analytic_jeans(
                a, b, c,
                return_ls_obj=True,
                **lsq_fitter_kwargs
            )

            return SIDMHaloSolution(
                outer_nfw, inner_soln,
                r_1, jeans_CSE_decomp,
            )

        # Now the case with a baryon profile
        # TODO allow tuning inner radius etc
        integrated_result, params =_baryons_solver.integrate_isothermal_region(
            N0, sigma_0, cross_section, halo_age, baryon_profile=baryon_profile
        )
        r_0 = params['r0']
        r_1 = params['r1']
        rho_0 = params['rho_0']
        rho_1 = params['rho_1']

        inner_soln = InnerIsothermal(
            cross_section, sigma_0, rho_0, halo_age,
            solution_interp=lambda x: integrated_result.sol(x)[0],
            rscale_interp=r_0,
            mag_interp=rho_0,
        )

        outer_nfw = OuterNFW.solve_from_boundary(r_1, rho_1, params['Menc'], z)

        a = (r_1 / outer_nfw.r_s).to(1).value
        b = (r_0 / outer_nfw.r_s).to(1).value
        c = (rho_0 / outer_nfw.rho_s).to(1).value

        jeans_CSE_decomp = decompose_integrated_jeans(
            lambda x: integrated_result.sol(x)[0], a, b, c,
            **lsq_fitter_kwargs
        )

        return SIDMHaloSolution(
            outer_nfw, inner_soln,
            r_1, jeans_CSE_decomp,
        )

    @property
    def cross_section(self):
        return self.isothermal_region.cross_section

    @property
    def sigma_0(self):
        return self.isothermal_region.sigma_0

    @property
    def rho_0(self):
        return self.isothermal_region.rho_0

    @property
    def halo(self):
        '''
        Returns the Colossus NFW halo object representing the outer 'skirt'
        '''
        return self.outer_nfw.halo

    @property
    def r_s(self):
        return self.outer_nfw.r_s

    @property
    def rho_s(self):
        return self.outer_nfw.rho_s

    def density_3d(self, r):
        '''
        Computes the 3D halo density at radius r in kpc using the CSE decomposition
        of this halo, in units of Msun / kpc3.
        '''
        r = require_units(r, 'kpc')
        x = (r / self.r_s).to(1).value
        return self.rho_s * self.cse_decomp.density_3d(x)

    def mass_enclosed_3d(self, r):
        '''
        Computes the mass enclosed within a radius r in kpc using the
        CSE decomposition of this halo.

        Units of Msun
        '''
        r = require_units(r, 'kpc')
        x = (r / self.r_s).to(1).value
        return (
            self.rho_s * self.r_s**3 * self.cse_decomp.mass_enc_3d(x)
        ).to('Msun')

    def projected_density_2d(self, r):
        r = require_units(r, 'kpc')
        x = (r / self.outer_nfw.r_s).to(1).value
        return (
            self.outer_nfw.r_s * self.outer_nfw.rho_s * self.cse_decomp.proj_density(x)
        ).to('Msun kpc-2')

    def to_lenstronomy(self, lens_cosmo: LensCosmo):
        '''
        Converts the CSE decomposition of this halo into a lenstronomy format,
        in order to compute lensing observables
        '''
        rs_arcsec = lens_cosmo.phys2arcsec_lens(self.r_s.to(u.Mpc).value)

        # Projected weights. Scale by r_s for Abel transform
        cse_a_list = (
            self.cse_decomp._weights * self.rho_s * self.r_s
        ).to('Msun / Mpc2').value
        # Scale to kappa
        cse_a_list = cse_a_list * rs_arcsec**3 / lens_cosmo.sigma_crit
        cse_s_list = self.cse_decomp._esses * rs_arcsec
        return ['CSE_PROD_AVG_SET'], [{'a_list': cse_a_list, 's_list': cse_s_list, 'q': 1}]
