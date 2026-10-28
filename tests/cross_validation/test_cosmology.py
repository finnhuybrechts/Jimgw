"""Cross-validation tests for cosmology functions against astropy.

This module compares Jim's JAX cosmology implementations with astropy's.
Jim assumes a flat LambdaCDM universe without radiation, so the astropy
reference is always ``FlatLambdaCDM(H0=H0, Om0=Om0, Tcmb0=0)``.

Requires:
    - astropy
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.utils import assert_all_finite

pytest.importorskip("astropy")

n_samples = 10000


class TestRedshiftFromLuminosityDistanceHighLevel:
    """High-level cross-validation tests comparing Jim cosmology functions to astropy."""

    def test_z_at_dL_oneway(self):
        """Test Jim's z_at_dL against astropy's z_at_value reference.

        1. Generate random Hubble constants, matter densities and luminosity distances.
        2. astropy: d_L + FlatLambdaCDM(H0, Om0, Tcmb0=0) → redshift z (reference).
        3. Jim: d_L + (H0, Om0) → redshift z — assert matches astropy.
        """
        import astropy.units as u
        from astropy.cosmology import FlatLambdaCDM, z_at_value

        from jimgw.core.single_event.cosmology import z_at_dL

        key = jax.random.key(42)
        subkeys = jax.random.split(key, 3)

        # Step 1: random cosmologies and luminosity distances
        H0 = np.array(
            jax.random.uniform(subkeys[0], (n_samples,), minval=40.0, maxval=90.0)
        )
        Om0 = np.array(
            jax.random.uniform(subkeys[1], (n_samples,), minval=0.1, maxval=1.0)
        )
        d_L = np.array(
            jax.random.uniform(subkeys[2], (n_samples,), minval=1.0, maxval=50000.0)
        )

        # Step 2: astropy reference, one root-find per sample
        astropy_z = []
        for i in range(n_samples):
            z = z_at_value(
                FlatLambdaCDM(H0=H0[i], Om0=Om0[i], Tcmb0=0).luminosity_distance,
                d_L[i]*u.Mpc, # astropy needs units
                ztol = 1e-12,
            ).value
            astropy_z.append(z)

        astropy_z = jnp.array(astropy_z)

        # Step 3: Jim matches astropy
        jim_z = jax.vmap(z_at_dL)(jnp.array(H0), jnp.array(d_L), jnp.array(Om0))

        # Check if two arrays are element-wise approximately equal within a tolerance
        assert jnp.allclose(jim_z, astropy_z, rtol=1e-5, atol=0.0), (  
            f"Jim and astropy z disagree; max relative error "
            f"{jnp.max(jnp.abs(jim_z / astropy_z - 1.0)):.2e}"
        )

        assert_all_finite(jim_z) # Assert all values in the array are finite 
        # (AssertionError if array contains non-finite values)