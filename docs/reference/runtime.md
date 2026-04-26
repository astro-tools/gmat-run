# Runtime bootstrap

Load `gmatpy` for a resolved [`GmatInstall`][gmat_run.GmatInstall].
[`bootstrap`][gmat_run.bootstrap] takes the install, ensures the API startup
file exists (generating it on first use), prepends the install's `bin/` to
`sys.path`, imports `gmatpy`, calls `gmat.Setup(...)`, and returns the
imported module.

[`Mission.load`][gmat_run.Mission.load] calls `bootstrap` for you. Reach for it
directly only if you are driving GMAT outside the [`Mission`][gmat_run.Mission]
abstraction.

`bootstrap` is idempotent within a single interpreter, but a second call
requesting a *different* install raises
[`GmatLoadError`][gmat_run.GmatLoadError] — see
[Known limitations → gmatpy single-init constraint](../known-limitations.md#gmatpy-single-init-constraint).

::: gmat_run.bootstrap
