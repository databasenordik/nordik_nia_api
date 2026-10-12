# Standardized research data

The four supported lists use First Name, Middle Name, and Last Name. Header capitalization
may differ in an input export; the reviewed copies use the same spelling throughout.
Source name cells and Name comments are authoritative. Normalization rebuilds names,
search tokens, and dates from these cells rather than feeding previous parser output back
into the parser. Compound given names and surnames stay in their declared columns.
Historical full-name parsing is limited to the one-time export revision.

Community values also come directly from the declared source column. A blank source
clears an older inferred value. Indigenous Community Name and First Nation Community
remain separate attributes. The three death lists share their source field catalog;
an empty standardized column is available with zero recorded values, rather than being
treated as an unsupported attribute.

| Date notation | Meaning |
| --- | --- |
| `1900-01-01` | Recorded day |
| `1900-06-00` | Recorded month; day unknown |
| `1900-00-00` | Recorded year; month and day unknown |
| Empty, `0`, `0000-00-00` | Unknown date |
| `0000-09-00` | September recorded; year and day unknown |
| `A \| B` | Alternatives: one is correct |
| `A & B` | Both are correct; separate events |
| `A : B` | Inclusive range, at the recorded precision |
| `1909-09-04-06` | Alternatives September 4 or September 6, 1909 |

The shorthand is interpreted in code and is preserved in source data. It is not rewritten
in the database. A range is one event. AND dates count as separate dated events; OR dates
are alternatives rather than repeat events. The `*_event_count` fields report the minimum
number of recorded dated events guaranteed across the alternative branches.

Filters return definite matches. Every OR branch must establish the condition; any real
AND event can establish a positive condition. Negative comparisons require every event to
be disjoint from the queried interval. Partial dates span their recorded precision: a year
is not assumed to mean January 1. Ranges must fully satisfy a positive condition. Unknown
year components cannot establish chronological comparisons. A date with some usable
components can be recorded even if its year is unknown.

Derived year/month/day projections return a component only when it is shared by all
listed dates and every possible day within a range. Matching endpoint months do not
establish a common month for a range spanning different years. Event-year filters use
the full date expression. An expression is not silently reduced to its first date for
sorting or statistics; unknown years are excluded from numeric year statistics.

Lengths of stay in days require exact admission and discharge dates. Overlap comparisons
use the latest possible admission and earliest possible discharge to establish guaranteed
presence. Multiple admission/discharge events are not paired by guessing.

## Potential access

Only an explicit boolean `true` or a trimmed, case-insensitive `TRUE` source cell releases
a Potential row. FALSE, missing values, blanks, `1`, and `yes` deny access. The authoritative
permission is the current raw row's ACCESS cell; cached normalized flags cannot release it.
The Potential gate remains enforced when its policy key is missing or changed.

PostgreSQL security-barrier views enforce this before returning row payloads. All runtime
operations, normalization reads, and metadata refreshes use those views. Potential catalog
responses omit cached recorded values and counts, which could otherwise survive a row's
revocation. Static field definitions and validation rules remain available.
The runtime role cannot read the underlying source tables. The in-memory gateway and CSV
loader apply the equivalent gate before storing/materializing rows. Potential history is
invalidated before loading summaries, assistant messages, or query frames when the current
release revision changes.

Reviewed Potential exports contain authorized rows only. Withheld source rows were neither
reviewed nor revised.

## Snapshot workflow

The reference exports and dump are snapshots. Testing restores them into isolated databases
and does not update the application's configured database.

1. `benchmarks.standardized_snapshot` audits authorized exports.
2. `benchmarks.standardize_exports` creates revised copies with Date comments and Source
   revision notes, plus a revision manifest. Original exports are preserved.
3. `benchmarks.load_standardized_snapshot` prepares an import that asserts the isolated
   target database name and includes authorized rows only.
4. Migrations 070–073 enforce access, register standardized fields, implement date semantics,
   and protect persisted Potential history. Run normalization after the standardized source
   data and registry are ready; complete both before serving queries on a deployment.
5. `python -m app.normalization.backfill_names` regenerates all four lists and refreshes
   catalog metadata. A second run should write no rows. Stored manual name overrides require
   the explicit `--use-name-overrides` option.
6. `standardized_qualification.py`, `standardized_security.py`, and `standardized_matrix.py`
   exercise the real PostgreSQL runtime role and configured AI provider in the isolated
   snapshot database. Keep answer/evidence artifacts with the ignored local review data.

An all-information projection uses `requested_fields=["*"]`; trusted code expands it to
the selected list's source attributes. Derived helpers and internal fields are excluded.
Planner and reviewer requirements use the same expansion, retaining the independent check.
