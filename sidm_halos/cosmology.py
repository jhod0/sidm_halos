from astropy import units as u
from astropy import cosmology as astropy_cosmo
from colossus.cosmology import cosmology as col_cosmo

try:
    col_cosmo.getCurrent()
except:
    print('No colossus cosmology set. Defaulting to Planck 2018')
    col_cosmo.setCosmology('planck18')


def set_astropy_cosmo(cosmo: astropy_cosmo.Cosmology, name, sigma8=None, ns=None, **kwargs):
    current = col_cosmo.getCurrent()
    if sigma8 is None:
        sigma8 = current.sigma8
    if ns is None:
        ns = current.ns
    return col_cosmo.fromAstropy(cosmo, sigma8=sigma8, ns=ns, cosmo_name=name, **kwargs)


def halo_age(z, formation_z=None):
    if formation_z is not None:
        tform = col_cosmo.current_cosmo.age(formation_z)
    else:
        tform = 0
    return (col_cosmo.current_cosmo.age(z) - tform) * u.Gyr


def h():
    return col_cosmo.current_cosmo.h
