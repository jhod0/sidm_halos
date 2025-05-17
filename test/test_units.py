import numpy as np
import pytest
from pytest import approx
from astropy import units as u
from sidm_halos import require_units


def test_units():
    assert require_units(10, 'kpc').unit == u.kpc
    assert require_units(35.0, 'kpc').unit == u.kpc
    assert require_units(0*u.kpc, 'kpc').unit == u.kpc
    assert require_units(1e6*u.m, 'kpc').unit == u.kpc
    assert require_units(1*u.mpc, 'kpc').unit == u.kpc

    assert require_units(4*u.lightyear, 'kpc').value == approx(1.22640558e-3)

    assert require_units(np.ones(3), 'Msun').unit == u.Msun
    assert require_units(np.ones((3, 3)), 'Gyr').unit == u.Gyr


    density = require_units(1e-6, 'kg cm-3')
    assert density.unit == u.Unit('kg cm-3')
    assert (density * (1*u.m)**3).unit == u.Unit('kg m3 cm-3')
    assert (density * (1*u.m)**3).to('kg').unit == u.Unit('kg')
    assert (density * (1*u.m)**3).to('kg').value == approx(1)


def test_unit_errors():
    with pytest.raises(ValueError):
        require_units('bogus', 'cm2 g')
    with pytest.raises(ValueError):
        require_units([{}, {}], 'cm2 g')
    with pytest.raises(ValueError):
        require_units({'test': 'dictionary'}, 'cm2 g')

    with pytest.raises(ValueError):
        require_units(123, 'bogus units')

    with pytest.raises(u.UnitConversionError):
        require_units(10*u.m, u.kg)
    with pytest.raises(u.UnitConversionError):
        require_units(10*u.Unit('kg m-3')*5*u.Unit('km3'), u.Mpc**3)
