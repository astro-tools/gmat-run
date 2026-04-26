# Mission

[`Mission`][gmat_run.Mission] is the main entry point. Construct one via
[`Mission.load`][gmat_run.Mission.load], which discovers a local GMAT install,
bootstraps `gmatpy`, and parses a `.script` into the live GMAT object graph.
Field access uses dotted-path keys against that graph
(`mission["Sat.SMA"]`); [`Mission.run`][gmat_run.Mission.run] executes the
mission sequence headlessly and returns a [`Results`](results.md).

See [Getting started → Quick start](../getting-started.md#quick-start) for the
full load → override → run → plot flow.

::: gmat_run.Mission
