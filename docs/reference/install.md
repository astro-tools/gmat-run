# Install discovery

Locate and validate a local GMAT install. The single public entry point is
[`locate_gmat`][gmat_run.locate_gmat]; it returns a
[`GmatInstall`][gmat_run.GmatInstall] describing the resolved install or raises
[`GmatNotFoundError`][gmat_run.GmatNotFoundError] listing every location it
checked.

[`Mission.load`][gmat_run.Mission.load] calls `locate_gmat` for you — reach for
the function directly only when you need the resolved paths *before* loading a
script (e.g. to print the version), or when you want to drive
[`bootstrap`][gmat_run.bootstrap] yourself. See
[Getting started → Quick start](../getting-started.md#quick-start) for the
typical flow, and
[Install GMAT](../install-gmat.md) for where the discovery looks on each
platform.

::: gmat_run.locate_gmat

::: gmat_run.GmatInstall
