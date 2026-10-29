"""Useful cosmological functions as in Astropy, but made JAX compatible.

All functions assume a flat LambdaCDM universe without radiation. Distances are
in Mpc and H0 is in km/s/Mpc. Every function works under ``jax.jit``,
``jax.grad`` and ``jax.vmap``, and accepts scalars or arrays for ``z``/``d_L``.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

c = 299792.458  #km/s


def efunc(z, Om0):
    """
    Dimensionless Hubble parameter, defined such that :math:`H(z) = H_0 E(z)`.

    Assumes a flat LambdaCDM universe (:math:`\\Omega_\\Lambda = 1 - \\Omega_m`)
    and neglects radiation.
    """
    # Might want to add radiation as well: + Or0 * (z + 1.0)**4
    return jnp.sqrt(Om0 * (z + 1.0) ** 3 + (1.0 - Om0))


def comoving_distance(H0, z, Om0, n_points=1000):
    """
    Comoving distance in Mpc.

    .. math:: d_C(z) = \\frac{c}{H_0} \\int_0^z \\frac{dz'}{E(z')}

    In a flat universe (k=0) the comoving transverse distance equals the
    comoving distance (see Baumann, Cosmology). The integral is evaluated with
    the trapezoid rule on ``n_points`` points between 0 and ``z``.
    """
    z_prime = jnp.linspace(0.0, z, n_points)  # shape (n_points, *jnp.shape(z))
    return c / H0 * jnp.trapezoid(1.0 / efunc(z_prime, Om0), z_prime, axis=0)


def luminosity_distance(H0, z, Om0, n_points=1000):
    """Luminosity distance in Mpc, :math:`d_L = (1 + z)\\, d_C`."""
    return (1.0 + z) * comoving_distance(H0, z, Om0, n_points)


def z_at_dL(H0, d_L, Om0=0.3111, n_points=1000, n_iter=10):
    """Redshift at which ``luminosity_distance(H0, z, Om0) == d_L``.

    JAX counterpart of ``astropy.cosmology.z_at_value(cosmo.luminosity_distance, d_L)``.
    Om0 default from table 2 of Planck18 (arXiv:1807.06209).

    Solves :math:`f(z) = d_L(z) - d_L = 0` with a fixed number of Newton steps,
    :math:`z \\leftarrow z - f(z) / f'(z)`, using the
    derivative of d_L with respect to z, computed with jax.jvp

    .. math:: \\frac{d d_L}{dz} = d_C(z) + (1 + z) \\frac{c}{H_0 E(z)}.

    The starting point is Hubble's law, :math:`z_0 = H_0 d_L / c`. Because
    :math:`d_L(z)` is increasing and convex, Newton converges monotonically from
    there.
    
    This has been tested to agree with astropy's z_at_value function up to 
    a relative error 1e-05, with d_L values between 1.0 and 50000.0 Mpc.
    """

    # Astropy uses: scipy.optimize import minimize_scalar
    # jax.scipy.optimize.minimize also exists, but doesn't seem reliable 
    # because it's still under maintenance


    #def newton_step(_, z):
    #    f = luminosity_distance(H0, z, Om0, n_points) - d_L
    #    df_dz = comoving_distance(H0, z, Om0, n_points) + (1.0 + z) * c / (H0 * efunc(z, Om0))
    #    return z - f / df_dz
    
    def newton_step(_, z):
        # Using jax.jvp rather than analytic expression for df_dz:
        # stays correct if efunc is altered (eg. when adding radiation)
        d_L_z, df_dz = jax.jvp(
            lambda zz: luminosity_distance(H0, zz, Om0, n_points), # creates a one-argument function, z ↦ d_L(z), 
            #with H0 and Om0 fixed at their current values. jax.jvp only differentiates wrt z when wrapped like this
            (z,), # point where you evaluate
            (jnp.ones_like(z),) # How fast luminosity distance function changes if you move the input from z in this (jnp.ones_like(z,),) direction
        )
        return z - (d_L_z - d_L) / df_dz

    # Add some kind of convergence criterion? So that there are enough steps to reach a certain tolerance?
    ## --> Use while loop instead of for loop???

    z0 = H0 * d_L / c # linear Hubble law as intitial guess
    return jax.lax.fori_loop(0, n_iter, newton_step, z0)