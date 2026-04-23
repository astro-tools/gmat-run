# Contributing to gmat-run

Thanks for your interest. This page is the one place to learn the workflow.

## Getting set up

```bash
git clone https://github.com/astro-tools/gmat-run.git
cd gmat-run
uv sync --all-groups
```

You also need a local GMAT install to run the integration tests. gmat-run does not
ship GMAT binaries. On Linux and Windows you can grab a build from
[SourceForge](https://sourceforge.net/projects/gmat/files/GMAT/); R2026a is the
primary development target.

## Branches and PRs

- One issue per branch. Branch names use a short prefix for type:
  - `feat/<slug>` — new capability, tied to a `type:feature` issue.
  - `fix/<slug>` — bug fix, tied to a `type:bug` issue.
  - `chore/<slug>` — infra / tooling / hygiene.
  - `docs/<slug>` — docs-only change.
- Open a PR against `main`. Put `Closes #<N>` in the PR description so the issue
  auto-closes on merge and the project board advances the card to Done.
- Squash-merge is the only merge method. The PR title becomes the squash commit
  subject — write it as a complete imperative sentence.

## Local checks before pushing

```bash
uv run pytest           # unit tests (integration tests are gated behind a marker)
uv run ruff check       # lint
uv run mypy             # types
```

CI re-runs all three on Ubuntu and Windows. Integration tests run in CI against a
cached GMAT install; you do not need to run them locally unless you are touching
the run/load path.

## Commit messages

Keep them short and imperative. One subject line, optional body.

- "Add `ReportFile` parser"
- "Fix leap-second off-by-one in `UTCGregorian` epoch parsing"

Do not include AI or tool attribution trailers in commits, PR titles, PR descriptions,
or comments — see the repo-level convention.

## Scope discipline

gmat-run's scope is deliberately narrow: run an existing `.script` and return
results as DataFrames. Before opening a feature issue, check the charter and the
existing issues to make sure the work belongs here and not in a future repo.

- **Building missions in Python →** [`gmatpyplus`](https://github.com/weasdown/gmatpyplus).
- **Generating `.script` text from Python →** [`pygmat`](https://pypi.org/project/pygmat/).
- **Parallel sweeps / Monte Carlo →** a future `astro-tools/gmat-sweep`.

## Questions

Open a [discussion](https://github.com/astro-tools/gmat-run/discussions) rather than
an issue for open-ended questions, usage help, or brainstorming.
