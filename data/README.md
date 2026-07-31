# Data Directory

This directory contains data files used by the ML Comment Service.

## Files

- **blocklist.txt** — List of toxic/offensive words used by the Aho-Corasick engine for fast-pass toxicity detection.

## Dataset Schema

Raw CSVs are **not** checked into version control. The evaluation dataset used for benchmarking has the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `comment_text` | `str` | The raw YouTube comment text |
| `sentiment` | `str` | Ground-truth label: `positive`, `neutral`, or `negative` |
| `is_toxic` | `int` | `1` = toxic, `0` = non-toxic |
| `is_spam` | `int` | `1` = spam, `0` = not spam |

### Source

Dataset was collected from public YouTube comments and manually labeled.
Use `scripts/csv_cleaner.py` to strip metadata columns before evaluation.
