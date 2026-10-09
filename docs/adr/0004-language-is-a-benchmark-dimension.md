# Language is a benchmark dimension, on a parallel corpus; English first

Status: superseded by ADR 0009 (English only).

Jev's vendor states English is its strongest language, while the original plan targeted Portuguese legal text. Rather than pick one, each Language is benchmarked with the same pipeline; English is built and published first, Portuguese follows. Both Languages use the same documents (a parallel EN/PT corpus, see ADR 0006), so scores can be compared directly across Languages, not only as the Jev-vs-baseline gap within each.

## Considered Options

- Different corpora per Language (US case law for English, Brazilian STJ for Portuguese): rejected for the first article, because corpus difficulty would confound the language effect. STJ remains a candidate follow-up.
