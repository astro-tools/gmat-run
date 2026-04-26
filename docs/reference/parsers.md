# Parsers

The `gmat_run.parsers` subpackage parses GMAT's output formats into
`pandas.DataFrame` objects. Each parser exposes a
`parse(path) -> pandas.DataFrame` function and depends only on the file
layout — no `gmatpy` import, no GMAT install required. This lets parser logic
be unit-tested against fixture files alone.

You normally do **not** call these directly:
[`Mission.run`][gmat_run.Mission.run] returns a [`Results`](results.md) whose
`.reports`, `.ephemerides`, and `.contacts` mappings already dispatch to the
right parser based on the file's content. Reach for these functions only when
you have a stand-alone GMAT output file you want to load without driving a
`Mission`, or when you are writing your own dispatch logic.

## ReportFile

::: gmat_run.parsers.reportfile.parse

## EphemerisFile (CCSDS-OEM, text)

::: gmat_run.parsers.ephemeris.parse

## EphemerisFile (STK-TimePosVel, text)

::: gmat_run.parsers.stk_ephemeris.parse

::: gmat_run.parsers.stk_ephemeris.is_stk_ephemeris

## EphemerisFile (SPK, binary)

::: gmat_run.parsers.spk.parse

::: gmat_run.parsers.spk.is_spk_ephemeris

## CCSDS-AEM attitude ephemeris

::: gmat_run.parsers.aem_ephemeris.parse

::: gmat_run.parsers.aem_ephemeris.is_aem_ephemeris

## ContactLocator

::: gmat_run.parsers.contact.parse

## Epoch promotion

::: gmat_run.parsers.epoch.promote_epochs
