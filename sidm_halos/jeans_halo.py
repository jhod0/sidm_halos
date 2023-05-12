


class SIDMHaloSolution:
    '''
    Represents a self-interacting dark matter halo with the semi-analytic Jeans method, using
    an outer NFW 'skirt' and inner isothermal region.
    '''
    def __init__(self, **kwargs):
        # should have:
        #   halo params:
        #       - M200m, c200m, z
        #       - rho_NFW, rs
        #   SIDM params:
        #       - cross_section (sigma/m)
        #       - r1, sigma_0 (v. disp)
        #       - rho_0
        #   CSE decomposition of solution
        pass

    @staticmethod
    def solve_outside_in(M200, c200, r1, z, baryon_profile=None):
        '''
        Constructs a SIDM halo via the 'outside-in' method: taking a known NFW
        M200/c200 and solving what the inner isothermal part of the halo should
        look like.
        '''
        raise NotImplementedError

    @staticmethod
    def solve_inside_out(cross_section, N0, sigma_0, z, baryon_profile=None):
        '''
        Constructs a SIDM halo via the 'inside-out' method, solving the inner
        isothermal profile and then finding an outer NFW halo which satisfies
        the boundary conditions.

        More efficient than the outside-in method.
        '''
        raise NotImplementedError
