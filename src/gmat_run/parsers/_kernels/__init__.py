"""Vendored SPICE kernels used by the SPK parser.

Currently:

* ``naif0012.tls`` — NAIF leap-seconds kernel, required for the
  ``TDB → UTC`` conversion in :mod:`gmat_run.parsers.spk`. Public domain;
  refreshed from ``naif.jpl.nasa.gov`` if NAIF publishes a new revision.
"""
