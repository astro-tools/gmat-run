# Mission summary

[`Mission.summary`][gmat_run.Mission.summary] returns a structured snapshot of
a loaded mission: resources grouped by type, declared output resources, and
the top-level mission-sequence outline. The same snapshot backs the notebook
reprs on [`Mission`][gmat_run.Mission] and [`Results`][gmat_run.Results] —
ending a cell on a bare `mission` reference renders the table inline instead
of the default `<gmat_run.mission.Mission object at 0x…>` form.

The walk is name-only and one level deep into branch commands. It does not
materialise field values (use `mission["Sat.SMA"]` for that) and it does not
render deeper nesting as a tree — a `Target` containing several `Vary`
commands each containing a `Propagate` shows the `Vary`s and a count of the
deeper descendants.

```python
from gmat_run import Mission

mission = Mission.load("flyby.script")
summary = mission.summary()

print(summary)
# MissionSummary('flyby.script')
#
# Resources
#   Spacecraft (1): Sat
#   ForceModel (1): FM
#   Propagator (1): Prop
#   ReportFile (1): RF
#
# Outputs
#   ReportFile: RF
#
# Mission sequence (2 commands)
#   1. Propagate — Propagate Prop(Sat) {Sat.ElapsedDays = 1};
#   2. Maneuver — Maneuver TOI(Sat);
```

## Schema

::: gmat_run.MissionSummary

::: gmat_run.ResourceGroup

::: gmat_run.CommandOutline
