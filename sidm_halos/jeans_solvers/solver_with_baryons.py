
'''
Solves the Jeans equations in the presence of a baryon matter component.
'''

from astropy import units as u
from astropy import constants
import numpy as np

# Could use solve_bvp - dedicated solver with boundaries
from scipy.integrate import solve_ivp, solve_bvp
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
        print(result.y)
        raise ValueError('no r1')
    # in Msun
    Menc = (4 * np.pi * (rho_0 * r0**3) * result.sol(r1_pred)[2])
    rho_1 = rho_0 * np.exp(result.sol(r1_pred))
    return result, {'rho_0': rho_0, 'r0': r0, 'rho_1': rho_1, 'r1': r1_pred*r0, 'Menc': Menc}


def solve_outside_in_as_bvp(r1, Menc, rho_1, halo_age, baryon_profile,
                            start_radius=1e-3,
                            N0_guess=10, sigma_0_guess=600,
                            abc=None,
                            **bvp_kwargs):
    '''
    Solves the 'outside-in' Jeans problem with baryons, using a boundary value solver
    '''
    r1 = require_units(r1, 'kpc')
    Menc = require_units(Menc, 'Msun')
    rho_1 = require_units(rho_1, 'Msun kpc-3')

    r1_kpc = r1.value
    Menc_msun = Menc.value
    rho_1_units = rho_1.value

    def fun(x, yvec, p):
        '''
        Integrand
        x = r / r_1     <- (N.B. NOT r / r_0 now: there is no r_0)
        p = log([N0, sigma0])
        yvec = [y, dydx, M(<x)]
        '''
        N0, sigma0 = np.exp(p)
        rho_0 = N0 * rho_1
        rho_0_units = rho_0.value
        const = (
            (4 * np.pi * constants.G * rho_0 * r1**2) / (sigma0 * u.Unit('km/s'))**2
        ).to(1).value

        y, dydx, M = yvec

        iso_density = np.exp(y)
        unitless_density = iso_density + (baryon_profile(x*r1)/rho_0).to(1).value
        # FIXME: This line is not robust to extreme conditions, and seems to have
        # overflow errors sometimes, which cause the bvp solver to fail.
        d2ydx2 = - 2 * dydx / x - const * unitless_density

        rho_r = rho_0_units * np.exp(y)
        # dMdx scaled by enclosed mass
        # x is unitless, scaled as x = r/r1
        dMdx = 4 * np.pi * x**2 * rho_r * r1_kpc**3 / Menc_msun

        return [dydx, d2ydx2, dMdx]

    def bc(yveca, yvecb, p):
        '''
        Boundary conditions of jeans equation
        '''
        logN0, logsigma0 = p

        ya, dydxa, Ma = yveca
        yb, dydxb, Mb = yvecb

        # Need 5 degrees of constraint (see note below)
        # Boundary conditions:
        #   - ya should be 0 (so rho(r=0) = rho_0)
        #   - dydx at a should be 0
        #   - M(<a) = 0
        #   - yb should correspond to rho_1 (therefore yb = - log (N0))
        #   - Mb should correspond to Menc, so 1 in units of Menc
        return [ya, dydxa, Ma, yb + logN0, np.log(Mb)]

    def fun_jac(x, yvec, p):
        '''
        Jacobian of `fun`, it's derivatives with respect to `yvec` and `p`.
        '''
        N0, sigma0 = np.exp(p)
        y, dydx, M = yvec
        _dydx, d2ydx2, dMdx = fun(x, yvec, p)
        rho_0 = N0 * rho_1
        const = (
            (4 * np.pi * constants.G * rho_0 * r1**2) / (sigma0 * u.Unit('km/s'))**2
        ).to(1).value

        iso_density = np.exp(y)
        unitless_density = -(2 * dydx / x + d2ydx2) / const

        zeros = np.zeros_like(x)
        ones = np.ones_like(x)

        df_dy = np.array([
            # Derivatives of dydx
            [zeros, ones, zeros],
            # Derivatives of d2ydx2
            [-const * np.exp(y), -2 / x, zeros],
            # Derivatives of dMdx
            [dMdx, zeros, zeros],
        ])
        df_dp = np.array([
            # Derivatives of dydx
            [zeros, zeros],
            # Derivatives of d2ydx2
            [-const * iso_density, 2 * const * unitless_density],
            # Derivatives of dMdx
            [dMdx, zeros],
        ])
        return df_dy, df_dp

    def bc_jac(yveca, yvecb, p):
        '''
        Jacobian of the boundary conditions, it's derivatives with respect to
        yveca, ybecb, and p.
        '''
        logN0, logsigma0 = p

        ya, dydxa, Ma = yveca
        yb, dydxb, Mb = yvecb

        dbc_dya = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0]
        ])
        dbc_dyb = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1/Mb]
        ])
        dbc_dp = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 0],
        ])
        return dbc_dya, dbc_dyb, dbc_dp

    x = np.logspace(
        np.log10(require_units(start_radius, 'kpc').value), np.log10(r1_kpc),
        101
    ) / r1_kpc

    # initial guess for y: 0 at x=0, -N0_guess at r1
    # TODO smarter initial conditions - make them the baryon-less solution
    if abc is None:
        y = -x * np.log(N0_guess)
        dydx = - np.ones_like(x) * np.log(N0_guess)
        M = x**3
    else:
        from .sidm_profiles import y_interp, dy_interp, mass_interp_
        a, b, c = abc
        y = y_interp(x * b / a)
        dydx = dy_interp(x * b / a) * (b / a)
        M = mass_interp_(x * b / a) * (a / b)**3
        M /= np.max(M)
        N0_guess = np.exp(-np.min(y))
    y_guess = np.vstack((y, dydx, M))

    # Dimensionality:
    #   - y(x) is a 3-vector (n = 3)
    #   - p is a 2-vector (k = 2)
    # According to solve_bvp docs, the boundary condition needs (n+k) degrees
    # of constraint for everything to be determined. Therefore bc needs to
    # return a 5d vector
    result = solve_bvp(
        fun, bc, x, y_guess, p=np.log([N0_guess, sigma_0_guess]),
        fun_jac=fun_jac,
        bc_jac=bc_jac,
        **bvp_kwargs,
    )

    if not result.success:
        print(result)
        msg = f'BVP solver failed ' \
              f'with message {result.message}'
        raise ValueError(msg)

    N0, sigma0 = np.exp(result.p)
    sigma0 *= u.Unit('km/s')
    rho_0 = rho_1 * N0
    cross_section = (1 / (4 * sigma0 / np.sqrt(np.pi) * rho_1 * halo_age)).to('cm2/g')

    params = {
        'N0': N0,
        'sigma_0': sigma0,
        'rho_0': rho_0,
        'rho_1': rho_1,
        'r1': r1,
        'r0': r1,
        'cross_section': cross_section,
    }

    return result, params


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
