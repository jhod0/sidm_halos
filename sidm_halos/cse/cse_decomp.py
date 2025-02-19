import numpy as np
import time
from scipy import optimize as opt

from .cse_3d import (rhoCSE_3d, rhoCSE_m_enc)
from .cse_2d import kappaCSE_2d
from ..jeans_solvers import sidm_profiles as _sidm_solved


class CSERepr:
    '''
    Represents a density profile as a sum of Cored Steep Ellipsoid (CSE) profiles

    TODO units?
    '''
    def __init__(self, A_s, esses, r_scale=1, magnitude=1):
        self._weights = np.array(A_s)
        self._esses = np.array(esses)
        assert self._weights.shape == self._esses.shape
        self._r_scale = r_scale
        self._magnitude = magnitude

    def proj_density(self, r):
        return self._r_scale * self._magnitude * kappaCSE_2d(
            r/self._r_scale, self._esses, weights=self._weights
        )

    def density_3d(self, r):
        return self._magnitude * rhoCSE_3d(
            r/self._r_scale, self._esses, weights=self._weights
        )

    def mass_enc_3d(self, r):
        return self._magnitude * rhoCSE_m_enc(
            r/self._r_scale, self._esses, weights=self._weights
        )


def decompose_cse(func, xs, esses, init_guess=None, sigma=1e-4, return_ls_obj=False, verbose=0,
                  fixed_weights=None, require_positive_weights=False, verbose=False, **lsq_kwargs) -> CSERepr:
    '''
    Decomposes a 3D density profile `func` into a sum of CSE profiles, with fixed size parameters `esses`.
    '''
    truth = func(xs)
    truth_zero_mask = truth == 0.0
    zero_offset = 1e-12
    esses = np.array(esses)

    if fixed_weights is not None:
        assert init_guess is not None, 'must provide initial guess if some weights are fixed'
        fixed_weights = np.array(fixed_weights)
        assert fixed_weights.shape == esses.shape
    else:
        fixed_weights = np.zeros_like(esses, dtype=bool)

    if init_guess is None:
        init_guess = esses**-4

    def residual(weights_sample):
        weights = np.empty_like(esses)
        weights[fixed_weights] = init_guess[fixed_weights]
        weights[~fixed_weights] = weights_sample
        calc = rhoCSE_3d(xs, esses, weights=weights)
        return (((calc + zero_offset*truth_zero_mask )/ (truth + zero_offset*truth_zero_mask)) - 1) / sigma

    def jacobian(weights_sample):
        weights = np.empty_like(esses)
        weights[fixed_weights] = init_guess[fixed_weights]
        weights[~fixed_weights] = weights_sample
        all_bases = rhoCSE_3d(xs, esses, weights=np.ones_like(weights), collapse=False)
        # The *weights is for the log-sampling
        return ((all_bases.T / (truth + zero_offset*truth_zero_mask)).T / sigma)[:, ~fixed_weights]

    # if init_guess is not None:
    #     logweights_guess = np.log(init_guess)

    # TODO ensure converged?
    if verbose > 0:
        start = time.time()
    if require_positive_weights:
        bounds = (0, np.inf)
    else:
        bounds = (-np.inf, np.inf)
    # print(init_guess)
    # print(f'init guess: {init_guess[~fixed_weights]}')
    x_scale = (esses**4)[~fixed_weights]
    # FIXME: try opt.nnls: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html#scipy.optimize.nnls
    # non-negative by construction. gets around having nonphysical negative components, even
    # if they give a globally good solution
    try:
        lsq_soln = opt.least_squares(
            residual, init_guess[~fixed_weights], jac=jacobian,
            x_scale=x_scale,
            verbose=verbose,
            bounds=bounds,
            **lsq_kwargs
        )
    except:
        init_resid = residual(init_guess[~fixed_weights])
        if verbose and not np.all(~np.isfinite(init_resid)):
            print('non-finite residual for x values:')
            xs_non_finite = xs[~np.isfinite(init_resid)]
            print(xs_non_finite)
            print('which have truth values:')
            print(func(xs_non_finite))
            print(xs[truth == 0.0])
        raise
    if verbose > 0:
        end = time.time()
        print(f'lsq took {end-start:.3f} seconds')

    weights_soln = np.empty_like(esses)
    weights_soln[fixed_weights] = init_guess[fixed_weights]
    weights_soln[~fixed_weights] = lsq_soln.x
    soln = CSERepr(weights_soln, esses)

    if return_ls_obj:
        return soln, lsq_soln
    return soln


def decompose_nfw_like(func, ess_max, return_ls_obj=False,
                       **kwargs):
    '''
    Decomposes an NFW-like density profile into a CSE expansion

    Takes the Oguri NFW CSE parameters as a starting point.

    :param func: the density profile to decompose. Should take arguments in units of r_s and return in units of rho_s
    '''
    xs = np.logspace(-6, 2, 251)
    fixed_weights = (NFWCSEDecomp._esses > ess_max)
    return decompose_cse(
        func, xs,
        NFWCSEDecomp._esses, NFWCSEDecomp._weights,
        fixed_weights=fixed_weights,
        return_ls_obj=return_ls_obj,
        **kwargs
    )


def decompose_analytic_jeans(a, b, c, **kwargs):
    return decompose_integrated_jeans(_sidm_solved.y, a, b, c, **kwargs)


def decompose_integrated_jeans(y_interp, a, b, c, **kwargs):
    '''
    Decomposes a profile consisting of an outer NFW 'skirt' and an inner
    interpolated solution into a CSE expansion.

    y_interp should be a function that takes x=(r/r_0) and returns
    ln(rho(r)/rho_0), where rho(0) = rho_0. So y(0) = 0 and y(>0) should
    be negative

    a = r1 / rs
    b = r0 / rs
    c = rho0 / rhoNFW
    '''
    def jeans(xs):
        answer = np.empty_like(xs)
        answer[xs < a] = c*np.exp(y_interp(xs[xs < a]/b))
        skirt_msk = xs >= a
        skirt_xs = xs[skirt_msk]
        answer[skirt_msk] = 1 / (skirt_xs * (1 + skirt_xs)**2)
        return answer

    if 'ess_max' not in kwargs:
        kwargs['ess_max'] = 10 * a

    return decompose_nfw_like(jeans, **kwargs)


_CSE_WEIGHTS_OGURI = np.array([
    [1.082411e-06, 1.648988e-18],
    [3.292868e-06, 3.646620e-17],
    [8.786566e-06, 6.274458e-16],
    [1.860019e-05, 3.459206e-15],
    [3.274231e-05, 2.457389e-14],
    [6.232485e-05, 1.059319e-13],
    [9.256333e-05, 4.211597e-13],
    [1.546762e-04, 1.142832e-12],
    [2.097321e-04, 4.391215e-12],
    [3.391140e-04, 1.556500e-11],
    [5.178790e-04, 6.951271e-11],
    [8.636736e-04, 3.147466e-10],
    [1.405152e-03, 1.379109e-09],
    [2.193855e-03, 3.829778e-09],
    [3.179572e-03, 1.384858e-08],
    [4.970987e-03, 5.370951e-08],
    [7.631970e-03, 1.804384e-07],
    [1.119413e-02, 5.788608e-07],
    [1.827267e-02, 3.205256e-06],
    [2.945251e-02, 1.102422e-05],
    [4.562723e-02, 4.093971e-05],
    [6.782509e-02, 1.282206e-04],
    [1.127751e-01, 7.995270e-04],
    [1.596987e-01, 4.575541e-04],
    [2.169469e-01, 5.013701e-03],
    [3.423835e-01, 1.403508e-02],
    [5.194527e-01, 5.230727e-02],
    [8.623185e-01, 1.898907e-01],
    [1.382737e+00, 3.643448e-01],
    [2.034929e+00, 7.203734e-01],
    [3.402979e+00, 1.717667e+00],
    [5.594276e+00, 2.217566e+00],
    [8.052345e+00, 3.187447e+00],
    [1.349045e+01, 8.194898e+00],
    [2.603825e+01, 1.765210e+01],
    [4.736823e+01, 1.974319e+01],
    [6.559320e+01, 2.783688e+01],
    [1.087932e+02, 4.482311e+01],
    [1.477673e+02, 5.598897e+01],
    [2.495341e+02, 1.426485e+02],
    [4.305999e+02, 2.279833e+02],
    [7.760206e+02, 5.401335e+02],
    [1.935749e+03, 1.775124e+03],
    [2.143057e+03, 9.743682e+02]]
)


NFWCSEDecomp = CSERepr(4*_CSE_WEIGHTS_OGURI[:, 1], _CSE_WEIGHTS_OGURI[:, 0])
