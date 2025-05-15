import numpy as np
import pytest
from sidm_halos import SIDMHaloSolution, SIDMSolutionError


def test_simple():
    halo_params = [
        (5e14, 5, 50, 0.3),
        (1e14, 3, 100, 0.5),
        (5e13, 10, 10, 0.6),
        (1e15, 10, 100, 0.2),
    ]
    for (M, c, r1, z) in halo_params:
        halo_soln = SIDMHaloSolution.solve_outside_in(
            M=M, c=c, r1=r1, z=z,
            mdef='200m',
        )
        halo_soln_repeat = SIDMHaloSolution.solve_inside_out(
            cross_section=halo_soln.cross_section,
            N0=halo_soln.N0,
            sigma_0=halo_soln.sidm_sigma_0,
            z=halo_soln.z,
            mdef='200m',
        )

        # Test @properties
        for h in [halo_soln, halo_soln_repeat]:
            h.mass
            h.concentration
            h.mdef
            h.z
            h.cross_section
            h.sidm_sigma_0
            h.r1
            h.nfw_Vmax
            h.halo_age

        # Test that recovered parameters are close
        # This is a 0.5% tolerance
        rtol = 5e-3
        assert np.isclose(halo_soln.mass, halo_soln_repeat.mass, rtol=rtol)
        assert np.isclose(halo_soln.concentration, halo_soln_repeat.concentration, rtol=rtol)
        assert np.isclose(halo_soln.r1, halo_soln_repeat.r1, rtol=rtol)
        assert np.isclose(halo_soln.cross_section, halo_soln_repeat.cross_section, rtol=rtol)
        assert np.isclose(halo_soln.sidm_sigma_0, halo_soln_repeat.sidm_sigma_0, rtol=rtol)
        assert np.isclose(halo_soln.nfw_Vmax, halo_soln_repeat.nfw_Vmax, rtol=rtol)
