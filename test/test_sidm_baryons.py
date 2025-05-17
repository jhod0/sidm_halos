import numpy as np
import pytest
from astropy import units as u
import warnings
from sidm_halos import SIDMHaloSolution, SIDMSolutionError, OuterNFW
from sidm_halos.baryon_profiles import DPIEProfile
from sidm_halos import require_units

from check_halo import check_halo


@pytest.mark.parametrize("M,ra,rs", [
    (1e12, 1*u.kpc, 30),
    (1e11, 5, 10*u.kpc),
    (3e10*u.Msun, 0.1, 50),
    (3e11*u.Msun, 3*u.kpc, 20*u.kpc),
])
def test_dpie(M, ra, rs):
    from_mass = DPIEProfile.from_mass_ra_rs(M, ra, rs)
    from_density = DPIEProfile.from_rho_0_ra_rs(
        from_mass.rho_0, from_mass.ra, from_mass.rs
    )
    assert np.isclose(from_mass.mass, from_density.mass)
    assert np.isclose(from_mass.rho_0, from_density.rho_0)


@pytest.mark.parametrize("Mbar,ra_bar,rs_bar", [
    (1e12, 0.01, 30),
    (5e11, 1.5, 10),
    (1e10, 10, 15),
    (1e10, 15, 100),
    (1e12, 15, 100),
    # A trivially tiny baryon profile - just to test
    (1e5, 0.01, 1),
])
@pytest.mark.parametrize("Mhalo,chalo,rhalo,zhalo", [
    (10**14.7, 9, 100*u.kpc, 0.3),
    (5e13, 5, 10, 0.7),
    (1e14, 3, 50, 0.5),
])
@pytest.mark.parametrize('mdef', ['200m', '200c', 'vir', '500m',])
def test_with_dPIE(Mbar, ra_bar, rs_bar, Mhalo, chalo, rhalo, zhalo, mdef):
    baryons = DPIEProfile.from_mass_ra_rs(
        Mbar, ra_bar, rs_bar,
    )
    halo_soln_baryon = SIDMHaloSolution.solve_outside_in(
        M=Mhalo, c=chalo, r1=rhalo, z=zhalo,
        baryon_profile=baryons, mdef=mdef,
    )
    check_halo(halo_soln_baryon)
    halo_soln_no_baryon = SIDMHaloSolution.solve_outside_in(
        M=Mhalo, c=chalo, r1=rhalo, z=zhalo,
        mdef=mdef,
    )
    check_halo(halo_soln_no_baryon)

    # Baryons make the halos hotter
    assert halo_soln_baryon.sidm_sigma_0 > halo_soln_no_baryon.sidm_sigma_0

    if Mbar > 1e10:
        with pytest.raises(SIDMSolutionError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # r1 close to 4 should fail in baryon case
                SIDMHaloSolution.solve_outside_in(
                    M=Mhalo, c=chalo, r1=halo_soln_baryon.r_s*3.9, z=zhalo,
                    baryon_profile=baryons, mdef=mdef,
                )
    with pytest.raises(SIDMSolutionError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # r1 > 4*rs fails for both baryon and no-baryon case
            SIDMHaloSolution.solve_outside_in(
                M=Mhalo, c=chalo, r1=halo_soln_baryon.r_s*4.1, z=zhalo,
                baryon_profile=baryons, mdef=mdef,
            )


def gaussian_density(r, M, sigma):
    r = require_units(r, 'kpc')
    M = require_units(M, 'Msun')
    sigma = require_units(sigma, 'kpc')
    norm = M / (np.sqrt(2 * np.pi) * sigma)**3
    return norm * np.exp(-r*r / (2 * sigma**2))


@pytest.mark.parametrize("Mbar,sigma_bar", [
    (1e12, 30),
    (5e11, 20),
    (1e10, 15),
    (1e10, 100),
    (1e12, 50),
    # A trivially tiny baryon profile - just to test
    (1e5, 1),
])
@pytest.mark.parametrize("Mhalo,chalo,rhalo,zhalo", [
    (10**14.7, 9, 100*u.kpc, 0.3),
    (5e13, 5, 10, 0.7),
    (1e14, 3, 50, 0.5),
])
@pytest.mark.parametrize('mdef', ['200m', '200c', 'vir', '500m',])
def test_with_gaussian(Mbar, sigma_bar, Mhalo, chalo, rhalo, zhalo, mdef):
    baryons = lambda r: gaussian_density(r, Mbar, sigma_bar)
    halo_soln_baryon = SIDMHaloSolution.solve_outside_in(
        M=Mhalo, c=chalo, r1=rhalo, z=zhalo,
        baryon_profile=baryons, mdef=mdef,
    )
    check_halo(halo_soln_baryon)
    halo_soln_no_baryon = SIDMHaloSolution.solve_outside_in(
        M=Mhalo, c=chalo, r1=rhalo, z=zhalo,
        mdef=mdef,
    )
    check_halo(halo_soln_no_baryon)

    # Baryons make the halos hotter
    assert halo_soln_baryon.sidm_sigma_0 > halo_soln_no_baryon.sidm_sigma_0

    if Mbar > 1e10:
        with pytest.raises(SIDMSolutionError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # r1 close to 4 should fail in baryon case
                SIDMHaloSolution.solve_outside_in(
                    M=Mhalo, c=chalo, r1=halo_soln_baryon.r_s*3.95, z=zhalo,
                    baryon_profile=baryons, mdef=mdef,
                )
    with pytest.raises(SIDMSolutionError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # r1 > 4*rs fails for both baryon and no-baryon case
            SIDMHaloSolution.solve_outside_in(
                M=Mhalo, c=chalo, r1=halo_soln_baryon.r_s*4.1, z=zhalo,
                baryon_profile=baryons, mdef=mdef,
            )
