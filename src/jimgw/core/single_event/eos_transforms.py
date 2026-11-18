"""Transforms that need jester (EOS + TOV). Importing this module requires jesterTOV and hicosmo."""

import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import jaxtyped
from jesterTOV.eos.spectral.spectral_decomposition import SpectralDecomposition_EOS_model
from jesterTOV.tov.gr import GRTOVSolver

from jimgw.core.single_event.cosmology import z_at_dL
from jimgw.core.single_event.transform_utils import Mc_q_to_m1_m2
from jimgw.core.transforms import NtoMTransform

import hicosmo
from hicosmo.models import LCDM


@jaxtyped(typechecker=typechecker)
class HubbleConstantEOSParametersToTidalDeformabilitiesTransform(NtoMTransform):
    """
    Transform the Hubble Constant H0 and the EOS parameters to the NS tidal deformabilities.
    
    First the Hubble Constant H0 and luminosity distance d_L are used to obtain redshift z.
    The redshift z is used to compute the source frame masses m_source_1, m_source_2 from
    the Chirp mass M_c and mass ratio q.
    Given the EOS (parameters) and source frame masses of the 2 NSs, the corresponding tidal
    deformabilities lambda_1 & lambda_2 can be computed.
    """

    def __repr__(self):
        return f"HubbleConstantEOSParametersToTidalDeformabilitiesTransform(name_mapping={self.name_mapping})"

    def __init__(self, Omega_m: float = 0.30966, Omega_b: float = 0.04897, Omega_k: float = 0.0, sigma8: float = 0.8102, n_s: float = 0.9665, ndat: int = 100, min_nsat: float = 0.75):
        # values from 	https://doi.org/10.1051/0004-6361/201833910 and https://docs.astropy.org/en/latest/api/astropy.cosmology.realizations.Planck18.html
        gamma_names = ["gamma_0", "gamma_1", "gamma_2", "gamma_3"]
        super().__init__((["H0"] + gamma_names, ["lambda_1", "lambda_2"]))

        # ENG reference values (Lindblom 2010); class attributes, so set once here
        # standard jester class attributes (p0, e0, xmax) are overwritten to satisfy the 
        # spectral decomposition parametrisation of ENG
        # might want to change this!!!! so that the attributes are not overwritten

        ##TODO: expand this class such that also MM EOS parametrisation etc can be used
        
        #G_cgs, c_cgs = 6.67430e-8, 2.99792458e10                                                         # commented out
        #SpectralDecomposition_EOS_model.p0_geom = 1.33e33 * G_cgs / c_cgs**4 * 1e4                       # commented out
        #SpectralDecomposition_EOS_model.e0_geom = 2.04e14 * G_cgs / c_cgs**2 * 1e4                       # commented out
        #SpectralDecomposition_EOS_model.xmax = 6.35                                                      # commented out
        eos = SpectralDecomposition_EOS_model(crust_name='DH')
        tov_solver = GRTOVSolver()

        def named_transform(x):
            #z = z_at_dL(x["H0"], x["d_L"], Om0)                                                          # commented out
            #c = 299792.458                                                                               # added 10/11 to test linear Hubble law
            #z = x["H0"] * x["d_L"] / c # linear Hubble law                                               # added 10/11 to test linear Hubble law
            cosmo = LCDM(H0=x["H0"], Omega_m=Omega_m, Omega_b=Omega_b, Omega_k=Omega_k, sigma8=sigma8, n_s=n_s)
            z = cosmo.dl_to_z(d_L=x["d_L"])
            m1, m2 = Mc_q_to_m1_m2(x["M_c"], x["q"])  # detector frame masses
            eos_data = eos.construct_eos({k: x[k] for k in gamma_names})
            
            # construct_family: Construct M-R-Λ curves by solving for multiple central pressures.
            # The central pressure grid spans from a minimum based on min_nsat to
            # the maximum pressure where the EOS remains causal (:math:`c_s^2 < 1`).
            # ndat: Number of points in central pressure grid (TOV eqs are solved for multiple central pressures)
            # min_nsat: Minimum central density (n_c) in units of saturation density (n_sat =0.16)
            # construct_family returns interpolated masses, radii and lambdas on a mass grid
            
            family = tov_solver.construct_family(eos_data, ndat=ndat, min_nsat=min_nsat)
            # when m_source > M_TOV, return 0 (\Lambda = 0, BH) because NSs with M > M_TOV are BHs
            lambda_1 = jnp.interp(m1 / (1 + z), family.masses, family.lambdas, right=0.0)
            lambda_2 = jnp.interp(m2 / (1 + z), family.masses, family.lambdas, right=0.0)
            return {"lambda_1": lambda_1, "lambda_2": lambda_2, "m1": m1, "m2": m2, "z":z}

        self.transform_func = named_transform