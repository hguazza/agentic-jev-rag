# Notebooks are thin consumers of an importable package

Status: superseded by ADR 0010.

PROJECT_CONTEXT.md plans notebooks first and a later refactor into FastAPI. Instead, all pipeline logic lives in an importable Python package from day one; notebooks only import, run and plot. This keeps the logic testable, keeps notebooks readable for the article, and turns the FastAPI phase into a thin HTTP layer rather than a rewrite.
