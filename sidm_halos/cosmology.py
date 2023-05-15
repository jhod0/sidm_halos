from astropy import units as u
from colossus.cosmology import cosmology as col_cosmo

try:
    col_cosmo.getCurrent()
except:
    print('No colossus cosmology set. Defaulting to Planck 2018')
    col_cosmo.setCosmology('planck18')


def halo_age(z, formation_z=None):
    if formation_z is not None:
        tform = col_cosmo.current_cosmo.age(formation_z)
    else:
        tform = 0
    return (col_cosmo.current_cosmo.age(z) - tform) * u.Gyr


def h():
    return col_cosmo.current_cosmo.h
