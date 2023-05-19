# SIDM Halos

Computes the density profiles of dark matter halos in the case of self-interacting
dark matter (SIDM).

Uses the isothermal jeans modeling formalism (see [Robertson 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.4610R/abstract) for a great overview).

Solutions are converted to a sum of Cored Steep Ellipsoid (CSE) profiles, which
can be fed to e.g. lenstronomy to give strong lensing observables.

## Use

```
from sidm_halos import SIDMHaloSolution
# Solve for a halo with M200m=5e14 Msun, c200m=6,
# and the crossover radius between NFW and isothermal at 75kpc
soln = SIDMHaloSolution.solve_outside_in(
    M=5e14, c=6, r1=75, z=0.3, mdef='200m'
)

# The SIDM cross-section and isothermal velocity
print(soln.cross_section)
print(soln.sigma_0)

rs = np.logspace(-2, 2, 1001)*soln.r_s
density = soln.density_3d(rs)

plt.plot(rs, density, label='SIDM')
plt.plot(rs, soln.outer_nfw.density_3d(rs), label='NFW')
plt.legend()
plt.loglog()
```

## Dependencies

- scipy
- numpy
- astropy
- colossus
- lenstronomy
