"""Transforms that need jester (EOS + TOV). Importing this module requires jesterTOV."""

import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import jaxtyped
from jesterTOV.eos.spectral.spectral_decomposition import SpectralDecomposition_EOS_model
from jesterTOV.tov.gr import GRTOVSolver

from jimgw.core.single_event.cosmology import z_at_dL
from jimgw.core.single_event.transform_utils import Mc_q_to_m1_m2
from jimgw.core.transforms import NtoMTransform


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

    def __init__(self, Om0: float = 0.30966, ndat: int = 100, min_nsat: float = 0.75):
        gamma_names = ["gamma_0", "gamma_1", "gamma_2", "gamma_3"]
        super().__init__((["H0"] + gamma_names, ["lambda_1", "lambda_2"]))

        # ENG reference values (Lindblom 2010); class attributes, so set once here
        G_cgs, c_cgs = 6.67430e-8, 2.99792458e10
        SpectralDecomposition_EOS_model.p0_geom = 1.33e33 * G_cgs / c_cgs**4 * 1e4
        SpectralDecomposition_EOS_model.e0_geom = 2.04e14 * G_cgs / c_cgs**2 * 1e4
        SpectralDecomposition_EOS_model.xmax = 6.35
        eos = SpectralDecomposition_EOS_model()
        tov_solver = GRTOVSolver()

        def named_transform(x):
            z = z_at_dL(x["H0"], x["d_L"], Om0)
            m1, m2 = Mc_q_to_m1_m2(x["M_c"], x["q"])  # detector frame masses
            eos_data = eos.construct_eos({k: x[k] for k in gamma_names})
            family = tov_solver.construct_family(eos_data, ndat=ndat, min_nsat=min_nsat)
            # Above M_TOV the object is not a NS for this EOS: treat as a BH (Lambda = 0)
            lambda_1 = jnp.interp(m1 / (1 + z), family.masses, family.lambdas, right=0.0)
            lambda_2 = jnp.interp(m2 / (1 + z), family.masses, family.lambdas, right=0.0)
            return {"lambda_1": lambda_1, "lambda_2": lambda_2}

        self.transform_func = named_transform