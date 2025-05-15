import numpy as np
import pytest
from astropy import units as u
from sidm_halos import SIDMHaloSolution, SIDMSolutionError, OuterNFW
from sidm_halos.baryon_profiles import DPIEProfile


def test_dpie():
    dpie_params = [
        (1e12, 1*u.kpc, 30),
        (1e11, 5, 10*u.kpc),
        (3e10*u.Msun, 0.1, 50),
        (3e11*u.Msun, 3*u.kpc, 20*u.kpc),
    ]
    for (m, ra, rs) in dpie_params:
        from_mass = DPIEProfile.from_mass_ra_rs(m, ra, rs)
        from_density = DPIEProfile.from_rho_0_ra_rs(
            from_mass.rho_0, from_mass.ra, from_mass.rs
        )
        assert np.isclose(from_mass.mass, from_density.mass)
        assert np.isclose(from_mass.rho_0, from_density.rho_0)

def test_with_baryons():
    baryons = DPIEProfile.from_mass_ra_rs(
        1e12, 0.01, 30
    )
    halo_soln = SIDMHaloSolution.solve_outside_in(
        M=10**14.7, c=8, r1=100*u.kpc, z=0.3,
        baryon_profile=baryons
    )
    with pytest.raises(SIDMSolutionError):
        # Try again at a larger r1 - it will fail
        halo_soln = SIDMHaloSolution.solve_outside_in(
            M=10**14.7, c=8, r1=halo_soln.r_s*3.9, z=0.3,
            baryon_profile=baryons,
        )
