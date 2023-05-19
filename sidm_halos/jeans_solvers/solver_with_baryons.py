
'''
Solves the Jeans equations in the presence of a baryon matter component.
'''

from astropy import units as u
from astropy import constants
import numpy as np

# Could use solve_bvp - dedicated solver with boundaries
from scipy.integrate import solve_ivp
from scipy import optimize as opt

from ..util import require_units


def jeans_integrand(x, yvec, baryons_unitless):
    '''
    Integrand for the isothermal equation
    '''
    y, dydx, M = yvec

    iso_density = np.exp(y)
    unitless_density = iso_density + baryons_unitless(x)
    d2ydx2 = - 2 * dydx / x - unitless_density
    dM = x**2 * iso_density
    return dydx, d2ydx2, dM


def integrate_isothermal_region(N0, sigma0, cross_section, halo_age, baryon_profile, start_kpc=0.01, **kwargs_solver):
    '''
    Integrates an isothermal jeans solution until r1 is reached.

    :param N0: The number of interactions over the lifetime of the halo at the halo center.
               Unitless. Should be strictly >= 1.
    :param sigma0: The isothermal velocity of the DM. km/s
    :param cross_section: Self-interaction cross section. cm2/g
    :param halo_age: Age of halo in Gyr.
    :param baryon_profile: The 3D density of baryons in the halo.
                           Should take an argument in kpc and return Msun/kpc3.


    :returns: 2-tuple of (integrator solution, dict of useful params)
    '''
    sigma0 = require_units(sigma0, 'km/s')
    cross_section = require_units(cross_section, 'cm2/g')
    halo_age = require_units(halo_age, 'Gyr')
    start_kpc = require_units(start_kpc, 'kpc')

    rho_0 = (N0 / (4 * sigma0 / np.sqrt(np.pi) * cross_section * halo_age)).to('Msun kpc-3')
    r0 = (sigma0 / np.sqrt(4 * np.pi * constants.G * rho_0)).to('kpc')

    # rho_0 = rho_0.value
    # r0 = r0.value

    def r1_event(x, yvec, *args):
        '''
        Event which triggers when r1 is reached
        '''
        y, dydx, dM = yvec
        return y + np.log(N0)
    r1_event.terminal = True

    start_x = (start_kpc / r0).to(1).value
    result = solve_ivp(
        jeans_integrand,
        # We want from r0/100 to 100*r0 (kind of arbitrary)
        [start_x, 1e5*start_x],
        # Initial values are (0, 0, 0) - log(rho/rho0) = 0, dydx = 0, dM=0
        [0, 0, 0],
        args=(lambda x: (baryon_profile(x*r0)/rho_0).to(1).value,),
        dense_output=True,
        events=[r1_event],
        vectorized=True,
        **kwargs_solver,
    )
    # There should always be an r1 if N0 > 1
    if result.t_events[0]:
        r1_pred = result.t_events[0][0]
    else:
        print(result.t_events)
        print(N0, sigma0)
        raise ValueError('no r1')
    # in Msun
    Menc = (4 * np.pi * (rho_0 * r0**3) * result.sol(r1_pred)[2])
    rho_1 = rho_0 * np.exp(result.sol(r1_pred))
    return result, {'rho_0': rho_0, 'r0': r0, 'rho_1': rho_1, 'r1': r1_pred*r0, 'Menc': Menc}


def solve_outside_in(r1, Menc, rho_1, halo_age, baryon_profile, N0_guess=10, sigma_0_guess=600):
    '''
    Solves the 'outside-in' Jeans problem with baryons.
    '''
    def func(x0):
        logN0, log_sigma_0 = x0
        N0 = (np.exp(logN0))
        sigma_0 = np.exp(log_sigma_0)
        sigma_0 *= u.Unit('km/s')
        cross_section = (1 / (4 * sigma_0 / np.sqrt(np.pi) * rho_1 * halo_age)).to('cm2/g')

        integ, params = integrate_isothermal_region(
            N0, sigma_0, cross_section, halo_age, baryon_profile,
            atol=1e-3, rtol=1e-2,
        )

        return np.log([
            (params['Menc'] / Menc).to(1).value,
            (params['r1'] / r1).to(1).value,
        ]).flatten()

    # Find the best fit
    # FIXME instead of using a root finder on an ivp integrator, we should
    # probably use a boundary value problem integrator
    result_root = opt.root(func, [np.log(N0_guess), np.log(sigma_0_guess)], tol=1e-3)

    N0_res, sigma_0_res = np.exp(result_root.x)
    sigma_0_res *= u.Unit('km/s')
    cross_section = (1 / (4 * sigma_0_res / np.sqrt(np.pi) * rho_1 * halo_age)).to('cm2/g')
    result_integrand, result_params = integrate_isothermal_region(N0_res, sigma_0_res, cross_section, halo_age, baryon_profile)

    result_params = dict(
        result_params,
        **dict(
            cross_section=cross_section,
            N0=N0_res,
            sigma_0=sigma_0_res,
        )
    )
    return result_integrand, result_params, result_root
