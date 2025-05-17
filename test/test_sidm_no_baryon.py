import numpy as np
import pytest
from sidm_halos import SIDMHaloSolution, SIDMSolutionError, OuterNFW

from check_halo import check_halo

@pytest.mark.parametrize("M,c,r1,z", [
    (5e14, 5, 50, 0.3),
    (1e14, 3, 100, 0.5),
    (5e13, 10, 10, 0.6),
    (1e15, 10, 100, 0.2),
    (1e15, 3, 200, 1),
    (1e15, 5, 15, 0.5),
    (1e13, 8, 5, 0.5),
    (1e13, 12, 5, 0.2),
    (1e15, 10, 100, 0.2),
    (1e15, 3, 200, 1),
    (1e15, 5, 15, 0.5),
    (1e15, 10, 100, 0.2),
    (1e15, 3, 200, 1),
    (1e15, 5, 15, 0.5),
])
@pytest.mark.parametrize('mdef', ['200m', '200c', 'vir'])
def test_simple(M, c, r1, z, mdef):
    halo_soln = SIDMHaloSolution.solve_outside_in(
        M=M, c=c, r1=r1, z=z,
        mdef=mdef,
    )
    halo_soln_repeat = SIDMHaloSolution.solve_inside_out(
        cross_section=halo_soln.cross_section,
        N0=halo_soln.N0,
        sigma_0=halo_soln.sidm_sigma_0,
        z=halo_soln.z,
        mdef=mdef,
    )

    # Test @properties
    for h in [halo_soln, halo_soln_repeat]:
        check_halo(h)
        assert h.mdef == mdef

    # Test that recovered parameters are close
    # This is a 0.01% tolerance
    rtol = 1e-4
    assert np.isclose(halo_soln.mass, halo_soln_repeat.mass, rtol=rtol)
    assert np.isclose(halo_soln.concentration, halo_soln_repeat.concentration, rtol=rtol)
    assert np.isclose(halo_soln.r1, halo_soln_repeat.r1, rtol=rtol)
    assert np.isclose(halo_soln.cross_section, halo_soln_repeat.cross_section, rtol=rtol)
    assert np.isclose(halo_soln.sidm_sigma_0, halo_soln_repeat.sidm_sigma_0, rtol=rtol)
    assert np.isclose(halo_soln.nfw_Vmax, halo_soln_repeat.nfw_Vmax, rtol=rtol)


@pytest.mark.parametrize("M,c,z,mdef", [
    (5e14, 5, 0.3, '200m'),
    (1e14, 3, 0.5, '200m'),
    (5e13, 10, 0.6, '200m'),
    (1e15, 10, 0.2, '200m'),
    (1e15, 3, 1, '200m'),
    (1e15, 5, 0.5, '200m'),
    (1e13, 8, 0.5, '200m'),
    (1e13, 12, 0.2, '200m'),
    (1e13, 12, 0.2, 'vir'),
    (1e13, 12, 0.2, '200c'),
    (1e13, 12, 0.2, '500m'),
])
def test_large_r1(M, c, z, mdef):
    outer_halo = OuterNFW(M=M, c=c, z=z, mdef=mdef)
    # Can't do r1/rs greater than 4
    with pytest.raises(SIDMSolutionError):
        SIDMHaloSolution.solve_outside_in(
            M=M, c=c, z=z, mdef=mdef, r1=4.5*outer_halo.r_s
        )
    # but this should work
    working_halo = SIDMHaloSolution.solve_outside_in(
        M=M, c=c, z=z, mdef=mdef, r1=3.9*outer_halo.r_s
    )
    check_halo(working_halo)
