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
        
        
class TestCosmologyRelativeErrorVsRedshift:
    """Relative error of Jim's z_at_dL against astropy as a function of z, 
    several n_points (amount of points in comoving distance trapezoid integral).

    The plot is only written when the ``cosmology_plot_directory`` environment variable
    points to a directory, so a normal test run produces no files.
    """

    H0_values = (40.0, 67.66, 100.0)
    #Om0_values = (0.1, 0.3111, 1.0)
    Om0 = 0.3111
    n_points_values = (100, 1000, 10000)
    z_grid = np.geomspace(1e-3, 6.0, 200)

    def test_relative_error_vs_z(self):
        """Compare z_at_dL to astropy for each n_points.

        The reference is the grid redshift itself: astropy computes d_L(z), Jim
        inverts it, and the recovered z is compared to the input z.
        """
        from astropy.cosmology import FlatLambdaCDM

        from jimgw.core.single_event.cosmology import z_at_dL

        errors = {}  # errors[H0][n_points] = relative error on z_grid
        for H0 in self.H0_values:
            cosmo = FlatLambdaCDM(H0=H0, Om0=self.Om0, Tcmb0=0)
            astropy_d_L = jnp.array(
                cosmo.luminosity_distance(self.z_grid).to_value("Mpc")
            )
            errors[H0] = {}
            for n_points in self.n_points_values:
                jim_z = z_at_dL(H0, astropy_d_L, self.Om0, n_points=n_points, n_iter=100)
                assert_all_finite(jim_z)
                errors[H0][n_points] = np.abs(np.asarray(jim_z) / self.z_grid - 1.0)

        plot_dir = os.environ.get("cosmology_plot_directory")
        if plot_dir:
            self._plot(errors, plot_dir)

        for H0, per_n_points in errors.items():
            assert np.max(per_n_points[1000]) < 1e-5, (
                f"H0={H0}, n_points=1000: max relative error "
                f"{np.max(per_n_points[1000]):.2e}"
            )
            # More points must give a smaller worst-case error
            max_errors = [np.max(per_n_points[n]) for n in self.n_points_values]
            assert all(np.diff(max_errors) < 0), (
                f"H0={H0}: error does not decrease with n_points: {max_errors}"
            )

    def _plot(self, errors, plot_dir):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            1, len(errors), figsize=(5 * len(errors), 4.5), sharey=True
        )
        for ax, (H0, per_n_points) in zip(axes, errors.items()):
            for n_points, err in per_n_points.items():
                # Exact agreement would vanish on a log axis, so floor it at 1e-17
                ax.loglog(
                    self.z_grid, np.maximum(err, 1e-17), label=f"n_points = {n_points}"
                )
            ax.set_title(f"H0 = {H0}")
            ax.set_xlabel("redshift z")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
        axes[0].set_ylabel("|z_Jim / z_astropy - 1|")
        fig.suptitle(
            f"z_at_dL vs astropy FlatLambdaCDM (Om0 = {self.Om0}, Tcmb0 = 0, n_iter = 100)"
        )
        fig.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        path = os.path.join(plot_dir, "z_at_dL_relative_error_vs_z_n_points.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"Saved {path}")