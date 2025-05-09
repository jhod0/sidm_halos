
'''
Solves the Jeans equations in the presence of a baryon matter component.
'''

from astropy import units as u
from astropy import constants
import numpy as np

# Could use solve_bvp - dedicated solver with boundaries
from scipy.integrate import solve_ivp, solve_bvp
from scipy import optimize as opt

from .error import SIDMSolutionError
from ..util import require_units


def solve_outside_in_as_bvp(r1, Menc, rho_1, halo_age, baryon_profile,
                            N0_guess, sigma_0_guess,
                            x_init, y_init,
                            start_radius=1e-4,
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

    # This represents the dimensionless constant at N0 = 1 and sigma0 = 100 km/s
    # This can then be scaled on each evaluation, without wasting time with units.
    # Seems to increase speed, simple tests show it going from 60-80ms -> 30-50ms
    # and 120-180ms -> 85-105ms
    scaled_const = (
        (4 * np.pi * constants.G * rho_1 * r1**2) / (100 * u.Unit('km/s'))**2
    ).to(1).value

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
        const = scaled_const * N0 / (sigma0 / 100)**2

        y, dydx, _M = yvec
        # fig, ax = plt.subplots(nrows=3)
        # ax[0].plot(x, y)
        # ax[1].plot(x, dydx)
        # ax[2].plot(x, M)

        iso_density = np.exp(y)
        unitless_density = iso_density + (baryon_profile(x*r1)/rho_0).to(1).value
        # FIXME: This line is not robust to extreme conditions, and seems to have
        # overflow errors sometimes, which cause the bvp solver to fail.
        #d2ydx2 = - 2 * dydx / x - const * unitless_density
        d2ydx2 = - const * unitless_density

        rho_r = rho_0_units * np.exp(y)
        # dMdx scaled by enclosed mass
        # x is unitless, scaled as x = r/r1
        dMdx = 4 * np.pi * x**2 * rho_r * r1_kpc**3 / Menc_msun

        return [dydx, d2ydx2, dMdx]

    def bc(yveca, yvecb, p):
        '''
        Boundary conditions of jeans equation
        '''
        logN0, _logsigma0 = p

        ya, dydxa, Ma = yveca
        yb, _dydxb, Mb = yvecb

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
        y, _dydx, _M = yvec
        _dydx, _d2ydx2, dMdx = fun(x, yvec, p)
        const = scaled_const * N0 / (sigma0 / 100)**2

        iso_density = np.exp(y)
        rho_0 = N0 * rho_1
        unitless_density = iso_density + (baryon_profile(x*r1)/rho_0).to(1).value

        zeros = np.zeros_like(x)
        ones = np.ones_like(x)

        df_dy = np.array([
            # Derivatives of dydx
            [zeros, ones, zeros],
            # Derivatives of d2ydx2
            # The singular term matrix S takes care of this
            # [-const * np.exp(y), -2 / x, zeros],
            # Derivatives of d2ydx2
            [-const * np.exp(y), zeros, zeros],
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
        _logN0, _logsigma0 = p

        _ya, _dydxa, _Ma = yveca
        _yb, _dydxb, Mb = yvecb

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

    # The singular term is d2y/dx2 = -2/x dy/dx
    S = np.array([
        [0, 0, 0],
        [0, -2, 0],
        [0, 0, 0],
    ])
    # Dimensionality:
    #   - y(x) is a 3-vector (n = 3)
    #   - p is a 2-vector (k = 2)
    # According to solve_bvp docs, the boundary condition needs (n+k) degrees
    # of constraint for everything to be determined. Therefore bc needs to
    # return a 5d vector
    result = solve_bvp(
        fun, bc, x_init, y_init, p=np.log([N0_guess, sigma_0_guess]),
        S=S,
        fun_jac=fun_jac,
        bc_jac=bc_jac,
        # verbose=2,
        **bvp_kwargs,
    )

    if not result.success:
        #print(result)
        msg = f'BVP solver failed ' \
              f'with message {result.message}'
        raise SIDMSolutionError(msg, fit_result=result)

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
