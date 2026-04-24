"""Parsers for GMAT output files.

Each parser exposes a ``parse(path) -> pandas.DataFrame`` function and depends
only on the file layout — no ``gmatpy`` import, no GMAT install required. This
lets parser logic be unit-tested against fixture files alone.
"""
