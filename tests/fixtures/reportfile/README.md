# `tests/fixtures/reportfile/` — provenance and regeneration

Real `ReportFile` output captured from stock GMAT `Ex_*.script` samples.
Used by `tests/test_parsers_reportfile_goldens.py` as golden files — parsing
them must produce exactly the DataFrames the tests assert. Any divergence
between a future GMAT release and the committed bytes surfaces as a loud test
failure, which is the point.

## Fixtures

| File | Source `.script` | Output extension at write time | Rows (data) | Columns | Notable |
|------|------------------|--------------------------------|-------------|---------|---------|
| `Ex_TLE_Propagation.txt` | `samples/Ex_TLE_Propagation.script` | `.txt` (unchanged) | 289 | 7 | UTCGregorian + Earth Cartesian state |
| `Ex_FiniteBurnParameters.txt` | `samples/Ex_FiniteBurnParameters.script` | `.txt` (renamed from `ReportFile1.txt`) | 121 | 11 | **A1ModJulian** numeric epoch; mixed float/int columns |
| `Ex_AnalyticMassProperties.txt` | `samples/Ex_AnalyticMassProperties.script` | `.txt` (renamed from `cm_moi_report.txt`) | 215 | 10 | Centre of mass + moment-of-inertia tensor |

File extensions are left as `.txt` — that is what the stock scripts write.
The parser does not key on extension, so this stays byte-faithful to GMAT's
own output.

## Capture details

- **GMAT version**: R2026a (`GmatConsole.exe` build date `Mar 26 2026`).
- **Install location at capture**: `D:\gmat-win-R2026a\` (WSL view: `/mnt/d/gmat-win-R2026a/`).
- **Capture date**: 2026-04-24.
- **Platform**: Windows 11 via WSL2 interop — `GmatConsole.exe` invoked
  directly as a Windows binary from a WSL shell. GMAT itself runs in the
  Windows process space; output lands on the Windows filesystem exactly as
  if run from `cmd.exe`. No Wine.

## Regeneration recipe

GMAT is deterministic: same script in, byte-identical report out. After a
GMAT upgrade these three lines reproduce all fixtures, and `git diff` will
flag any format drift the parser needs to learn.

```bash
cd /mnt/d/gmat-win-R2026a/bin

rm -f ../output/Ex_TLE_Propagation.txt \
      ../output/ReportFile1.txt \
      ../output/cm_moi_report.txt

./GmatConsole.exe --run ../samples/Ex_TLE_Propagation.script
./GmatConsole.exe --run ../samples/Ex_FiniteBurnParameters.script
./GmatConsole.exe --run ../samples/Ex_AnalyticMassProperties.script

cp ../output/Ex_TLE_Propagation.txt    <repo>/tests/fixtures/reportfile/Ex_TLE_Propagation.txt
cp ../output/ReportFile1.txt           <repo>/tests/fixtures/reportfile/Ex_FiniteBurnParameters.txt
cp ../output/cm_moi_report.txt         <repo>/tests/fixtures/reportfile/Ex_AnalyticMassProperties.txt
```

Output path is governed by `bin/gmat_startup_file.txt` (`OUTPUT_PATH = ../output/`).

## Licence

GMAT is Apache-2.0. Small captures of its text output embedded in this
MIT-licensed test suite are fine.
