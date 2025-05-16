import numpy as np
import pytest
from astropy import units as u
from sidm_halos import SIDMHaloSolution, SIDMSolutionError, OuterNFW

def check_halo(halo: SIDMHaloSolution):
    '''
    Checks some simple basics about an SIDM halo
    '''
    # First - just check all these methods and properties work
    repr(halo)
    halo.mass
    halo.concentration
    halo.mdef
    halo.r_s
    halo.rho_s
    halo.z
    halo.cross_section
    halo.sidm_sigma_0
    halo.N0
    halo.r1
    halo.nfw_Vmax
    halo.halo_age

    assert halo.N0 > 1
    assert halo.cross_section > 0*u.Unit('cm2/g')
    assert halo.sidm_sigma_0 > 0*u.Unit('km/s')

    # Check boundary conditions are satisfied
    rho_1 = halo.outer_nfw.density_3d(halo.r1)
    assert np.isclose(rho_1, halo.isothermal_region.density_3d(halo.r1))
    assert np.isclose(halo.N0*rho_1, halo.rho_0)
    # This is the analytic jeans boundary condition, see Robertson 2021 eqn. 3
    assert np.isclose(rho_1 * halo.cross_section * halo.sidm_sigma_0 * halo.halo_age * 4 / np.sqrt(np.pi), 1)
    # TODO should also check M(<r1) condition, but I don't seem to save the M(enc) relation for sidm

    # We don't expect the NFW and SIDM to be identical, but if they're more than a factor of 2 off
    # that's alarming
    assert 0.5 < halo.sidm_sigma_0/halo.nfw_sigma_0(halo.r1) < 2

    # This check that the CSE version approximates the true density well enough
    # Allow a 5% tolerance
    rtol_cse = 5e-2
    assert np.isclose(rho_1, halo.density_3d(halo.r1), rtol=rtol_cse)
    assert np.isclose(halo.rho_0, halo.density_3d(0), rtol=rtol_cse)
    assert np.isclose(halo.mass_enclosed_3d(halo.r1), halo.outer_nfw.mass_enclosed_3d(halo.r1), rtol=rtol_cse)
