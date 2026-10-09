# EU law parallel corpus (CJEU judgments + EU legislation), not Brazilian law

Status: superseded by ADR 0009 (parked with the legal benchmark).

The project began as a Brazilian legal PoC, but ADR 0004 needs the same documents in English and Portuguese. CJEU judgments (via the EU Publications Office Cellar/SPARQL, since eur-lex.europa.eu blocks bots) and EU legislation (MultiEURLEX) both exist in EN and PT with metadata, giving two Collections × two Languages. The trade-off: the Portuguese is European Portuguese and the law is EU law, not Brazilian. Brazilian STJ decisions are deferred to a follow-up run.

## Considered Options

- Caselaw Access Project (US) for English + STJ for Portuguese: different corpora per Language.
- CourtListener API: rate limits (125 requests/day) too low to build a corpus.
- UK Find Case Law: requires a separate licence for computational analysis.
