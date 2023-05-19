# SIDM Halos

Computes the density profiles of dark matter halos in the case of self-interacting
dark matter (SIDM).

Uses the isothermal jeans modeling formalism (see [Robertson 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.4610R/abstract) for a great overview).

Solutions are converted to a sum of Cored Steep Ellipsoid (CSE) profiles, which
can be fed to e.g. lenstronomy to give strong lensing observables.

## Dependencies

- scipy
- numpy
- astropy
- colossus
- lenstronomy
