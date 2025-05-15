#!/usr/bin/env python

'''
Computes Halo Mass Profiles for Self-Interacting Dark Matter.

Uses the semi-analytic Jeans approximation method, described in depth in
Robertson 2021: https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.4610R/abstract
'''



from astropy import units as u
import numpy as np

from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import root

import warnings


# ## Analytic Solution
#
# As mentioned above, for x close to zero we have an exact solution. Below a
# small threshold, we will approx y(x)/y'(x) with these solutions, above that threshold we will use odeint.


def y_approx(x):
    return np.log(1 - np.tanh(x/np.sqrt(6))**2)


def dy_approx(x):
    return -np.sqrt(2/3)*np.tanh(x/np.sqrt(6))


def ddy_approx(x):
    return -1/(3*np.cosh(x/np.sqrt(6)))


def _jeans_function_integrand(y, x):
    y, dy = y
    ddy = -(np.exp(y) + 2*dy / x)
    return np.array([dy, ddy])


xspan = np.logspace(-4, 2, 100_000)
start_x = (xspan[0])
solution = odeint(_jeans_function_integrand,
                  y0=[y_approx(start_x), dy_approx(start_x)],
                  t=xspan)

y_interp = interp1d(xspan, solution[:, 0], fill_value='extrapolate')
dy_interp = interp1d(xspan, solution[:, 1], fill_value='extrapolate')
x_from_y_interp = interp1d(solution[:, 0], xspan)


def y(x):
    '''
    Interpolate the reduced Jean's method function y(x).

    See Robertson 2021, equation 9.
    '''
    scalar = isinstance(x, (int, float))

    output = np.empty_like(x)
    x = np.array(x)
    output[x <= start_x] = y_approx(x[x <= start_x])
    output[x > start_x] = y_interp(x[x > start_x])

    if scalar:
        return output[()]
    return output


# ## Mass enclosed
#
# The boundary condidions we need to match are:
#
# - $\rho(r_0)$ - density at transition
# - $M(<r_1)$ - total mass within transition radius
#
# $$
# \begin{align}
#     M(<r_1) &= 4 \pi \int_0^{r_1} r^2 \rho(r) dr \\
#     &= 4 \pi \rho_0 \int_0^{r_1} r^2 e^{y(r_1/r_0)} dr \\
#     &= 4 \pi \rho_0 r_0^3 \int_0^{r_1/r_0} x^2 e^{y(x)} dx
# \end{align}
# $$
# In[5]:


def mass_integrand(y_, x):
    return x*x*np.exp(y(x))


xs_mass = np.concatenate([[0.0], np.logspace(-4, 2, 50_000)])

mass_integrated = odeint(mass_integrand, y0=[0], t=xs_mass).flatten()
mass_interp_ = interp1d(xs_mass, mass_integrated, fill_value='extrapolate')


def mass_interp(x):
    if x < xs_mass.min():
        return 0
    elif x > xs_mass.max():
        return mass_interp_(xs_mass.max())*(x/xs_mass.max())**2
    return mass_interp_(x)


# ## Boundary Condition Matching
#
# The whole profile will be defined piecewise by:
#
# $$
# \rho_{\text{jeans}}(r) = \begin{cases}
#           \rho_{\text{ISO}}(r) \quad &\text{if} \, x < r_1 \\
#           \rho_{\text{NFW}}(r) \quad &\text{if} \, x > r_1 \\
#      \end{cases}
# $$
#
# Where $\rho_{\text{ISO}}(r)$ is the isothermal profile described and computed above, and $\rho_{\text{NFS}}(r)$ is the standard NFW profile for the same halo. The transition radius $r_1$ is derived below.
#
# We have two parameters in $\rho_{\text{ISO}}$: $\rho_0$ and $\sigma_0$. We also have two boundary conditions which will let us pick those paramters:
#
# 1) $\rho_{\text{ISO}}(r_1) = \rho_{\text{NFW}}(r_1)$
#
# 2) $M_{\text{ISO}}(<r_1) = M_{\text{NFW}}(<r_1)$
#
# These should both be easy to find with the computations.
#
# $$ M_{\text{NFW}}(<r) = 4 \pi \rho_{\text{NFW}} r_s^3\Big(\frac{-r}{r + r_s} + \log(\frac{r + r_s}{r})\Big) $$
#
# The transition boundary is defined by:
#
# $$
# \frac{<\sigma v_{rel}>}{m} \rho_{dm}(r_1) t_{\text{age}} = 1 \\
# \frac{\rho_{NFW}}{(r_1/r_s)(1 + r_1/r_s)^2} = \frac{m}{<\sigma v_{rel}> t_{\text{age}}}
# $$
#
# There is an analytical soln, but it's a bit crazy. Let $a = \frac{m}{<\sigma v_{rel}> t_{\text{age}} \rho_{NFW}} $,
#
# $$
# r_1 = \frac{r_s}{6}\Big[-4 - \frac{2a}{\big(-\frac{27a^2}{2} - a^3 + 3/2\sqrt{3} a^2 \sqrt{27 + 4a}\big)^{1/3}} - \frac{2^{2/3}\big(-27a^2 - 2a^3 + 3\sqrt{3} a^2 \sqrt{27 + 4a}\big)^{1/3}}{a} \Big]
# $$


#def solve_nfw(a):
#    '''
#    Inverts the unitless NFW profile:
#
#        f(x) = 1 / (x (1 + x)^2)
#
#    I.e., gives x such that f(x) = a.
#    '''
#    term = 27*a**2 + 2*a**3 - 3*a**2*np.sqrt(3*(27 + 4*a))
#    return (
#        -4
#        + 2*a/((term/2)**(1/3))
#        + 2**(2/3)*term**(1/3) / a
#    ) / 6


# ### Unitless Jeans method solver
#
# Let's try to reparameterize this in a unitless sense. This way we will only need to know the ratio of the truncation radius $r_1$ to the NFW scale radius $r_s$.
#
# Let $a = \frac{r_1}{r_s}$, $b = \frac{r_0}{r_s}$, and $c = \frac{\rho_{\text{iso}}}{\rho_{\text{nfw}}}$. Given $a$, we use the following two conditions to constrain $b$ and $c$.
#
# 1. Enclosed mass
#
# $$
# \begin{align}
# M_{enc}^{iso}(<r_1) &= M_{enc}^{nfw}(<r_1) \\
# 4 \pi \rho_{\text{iso}} r_0^3 \int_0^{r_1/r_0} x^2 e^{y(x)} dx &= 4 \pi \rho_{\text{nfw}} r_s^3 \int_0^{r_1/r_s} \frac{x^2}{x(1 + x)^2} dx \\
# \frac{\rho_{\text{iso}}}{\rho_{\text{nfw}}} \Big(\frac{r_0}{r_s}\Big)^3 \int_0^{(r_1/r_s)(r_s/r_0)} x^2 e^{y(x) }dx &= \int_0^{r_1/r_s}\frac{x^2}{x(1 + x)^2} dx \\
# \end{align}
# $$
#
#
# $$
# \begin{align}
#     c b^3 \, \int_0^{\frac{a}{b}} x^2 e^{y(x)}dx &= \int_0^{a}\frac{x}{(1 + x)^2} dx \\
#     &= \frac{-a}{1 + a} + \log(1 + a)
# \end{align}
# $$
#
# 2. Density at boundary
#
# This is even easier:
#
# $$
# \begin{align}
#     \rho_{\text{iso}} e^{y(r_1/r_0)} &= \frac{\rho_{\text{nfw}}}{(r_1/r_s)(1 + r_1/r_s)^2} \\
#     c e^{y(\frac{a}{b})} &= \big[a(1 + a)^2\big]^{-1}
# \end{align}
# $$

# In[7]:


def _unitless_jeans_residual(x, a):
    '''
    Residual function: necessary for solving Jeans method boundary conditions.

    Only used as input to root-finding algorithm.
    '''
    log_b, log_c = x
    b, c = np.exp([log_b, log_c])

    # Enclosed mass
    exp_mass = -a / (1 + a) + np.log(1 + a)
    actual_mass = c * b**3 * mass_interp(a/b)

    # Density
    exp_density = 1/(a * (1 + a)**2)
    actual_density = c * np.exp(y(a / b))

    # These checks are to avoid divide-by-zero errors
    actual_density_zero = np.abs(actual_density) < 1e-8
    actual_mass_zero = np.abs(actual_mass) < 1e-8
    if not (actual_density_zero or actual_mass_zero):
        return 10*np.log([exp_mass / actual_mass, exp_density / actual_density])
    if actual_density_zero and actual_mass_zero:
        return 10 * np.array([exp_mass, exp_density])
    if actual_mass_zero and not actual_density_zero:
        return 10 * np.array([exp_mass, np.log(exp_density / actual_density)])
    assert actual_density_zero and not actual_mass_zero
    return 10 * np.array([np.log(exp_mass / actual_mass), exp_density])


#def _unitless_jeans_jacobian(x, a):
#    log_b, log_c = x
#    b, c = np.exp([log_b, log_c])
#
#    y_ = y(a/b)
#    dy_ = dy_interp(a/b)
#
#    # Derivatives:
#    #   df0/dx0    df0/dx1
#    #   df1/dx0    df1/dx1
#    return np.array([
#        [10 * (a/b)**3 * np.exp(y_) / mass_interp(a/b), -10],
#        [10 * dy_ * (a/b), -10]
#    ])


def solve_unitless_jeans(a, guess=None):
    '''
    Solves the boundary conditions between the NFW profile and isothermal Jeans
    profile.

    :param a: The crossover between NFW and isothermal profiles, in units of
              NFW scale radius. a = r1 / rs

    :returns: 2-tuple of (b = r0/rs, c = rho_iso / rho_NFW)
    '''
    if guess is None:
        # In log space - this means r0 = rs and rho_iso = rho_NFW
        x0 = [0.0, 0.0]
    else:
        x0 = np.log(guess)

    # Seems to actually run SLOWER when we use analytic jacobian :/
    soln = root(_unitless_jeans_residual,
                # jac=_unitless_jeans_jacobian,
                method='lm',
                x0=x0, args=(a,),
                tol=1e-14,
                options=dict(ftol=1e-14))

    if not soln.success:
        print(soln)
        raise ValueError(f'Couldnt converge for a = {a}')

    b, c = np.exp(soln.x)
    x = a / b
    if x > xs_mass[-1]:
        warnings.warn(
            f'Jeans solution (b, c) = ({b:.2e}, {c:.2e})'
            f' for a = {a:.2e} is out of bounds of mass interpolation'
        )
    if x > xspan[-1]:
        warnings.warn(
            f'Jeans solution (b, c) = ({b:.2e}, {c:.2e})'
            f' for a = {a:.2e} is out of bounds of Jeans equation solution'
            ' interpolation'
        )

    return b, c


# Solve boundary conditions on a large grid of a values
# That way - we will have high-quality initial guesses for (b, c)
# TODO this breaks at a >= 4: the solutions no longer work. Get rid of them
a_range = np.logspace(-2, np.log10(4), 3_000)
solved_b_c = []
guess = None
for a in a_range:
    try:
        soln_b_c = solve_unitless_jeans(a, guess=guess)
        solved_b_c.append(soln_b_c)
        # Keep this as the guess for next time
        guess = soln_b_c
    except ValueError as err:
        print('jeans solving failed at a =', a)
        print(err)
        solved_b_c.append([0.0, 0.0])

solved_b_c = np.array(solved_b_c)

_interp_b_guess = interp1d(a_range, solved_b_c[:, 0], bounds_error=True)
_interp_c_guess = interp1d(a_range, solved_b_c[:, 1], bounds_error=True)

_interp_a_from_a_over_b = interp1d(a_range / solved_b_c[:, 0], a_range)


def guess_b_c(a):
    try:
        b_guess = _interp_b_guess(a)
        c_guess = _interp_c_guess(a)
        return b_guess, c_guess
    except ValueError:
        if a < 1e-2:
            return a / 3, 3 / a
        if a > a_range[-1]:
            raise ValueError(f'a too high {a:.2e}, no physical solution')


#def unitless_jeans_profile(xs, a, b, c):
#    '''
#    Computes the Jeans method profile: NFW for x > a,
#    isothermal jeans for x < a.
#    '''
#    # xs are r/rs
#    output = np.empty_like(xs)
#    output[xs >= a] = 1/(xs[xs >= a] * (1 + xs[xs >= a])**2)
#    output[xs < a] = (
#        c*np.exp(y(xs[xs < a]/b))
#    )
#    return output
