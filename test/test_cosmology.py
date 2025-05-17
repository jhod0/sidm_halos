import numpy as np
import pytest
from astropy import units as u
import warnings
from sidm_halos import SIDMHaloSolution, SIDMSolutionError, OuterNFW
from sidm_halos.baryon_profiles import DPIEProfile
from sidm_halos import require_units
from sidm_halos import cosmology

from check_halo import check_halo


@pytest.mark.skip(reason='not implemented')
def test_cosmology():
    # TODO write tests using different cosmologies
    raise NotImplemented
