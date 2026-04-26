# Exceptions

Every exception gmat-run raises inherits from
[`GmatError`][gmat_run.GmatError], so downstream code can branch on failure
mode without string-parsing GMAT's log output. Leaf classes carry the payload
relevant to their failure mode as attributes (`attempts`, `log`, `path`,
`value`), never only as the formatted message.

::: gmat_run.GmatError

::: gmat_run.GmatNotFoundError

::: gmat_run.GmatLoadError

::: gmat_run.GmatRunError

::: gmat_run.GmatOutputParseError

::: gmat_run.GmatFieldError
