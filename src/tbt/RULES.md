# RULES.md — how we build the TBT agent (so we never lose the plot again)

Six rules. Each is the fix for a specific, documented way this project already failed — traceable in git:
the basal-ganglia orphaning (wired at `3ef5bb6`, silently un-wired by the `7c09cec` "collapse", never re-wired), the
137→80 silent test erosion across rewrites, four conflicting "single source of truth" docs, and three near-reinventions
of already-existing live code in a single session.

**The code + the passing tests are the ONLY source of truth. Everything else is history.**

## 1. The code is the map — `STATUS.md` is derived from it, never the reverse.
`STATUS.md` lists only what is actually wired, and must match the code. No narrative/plan/design doc ever outranks the
import graph. Superseded plans live under `Legacy - DO NOT USE OR IMPORT!/` and (going forward) `history/` — reference
only, never trusted over the code. When a doc and the code disagree, the code wins and the doc is fixed immediately.

## 2. "Unwired" is a bug, not a state. — the basal-ganglia rule.
Every module in `src/tbt/` is either reachable from the live entry point (`agent.py`) or on an explicit `STANDALONE`
allowlist with a written reason. A **reachability test** enforces this: the instant something is un-wired, the suite goes
RED and you must re-wire it or delete it. **No "defer the wiring to a later phase"** — that exact deferral is what killed
the basal ganglia, the thalamus, and the forward model.

## 3. Integration is "done." Built-in-isolation is a half-step, explicitly NOT done.
A mechanism counts only when it is WIRED into the live loop AND exercised by an end-to-end test that runs THROUGH the
loop. A unit test passing in a vacuum (e.g. the old `sequence.py`, "built + tested" for a week doing nothing) is not done.
Acceptance is always **"the agent plays more than it did before,"** never "the class passes its own test."

## 4. Build VERTICAL, never horizontal.
Always keep a runnable loop that wins SOME trivial task end-to-end (perceive → predict → plan → act → win). Every
increment thickens that one loop. Never build a whole layer in isolation to wire up "later" — that is the root cause of
the graveyard of built-but-unwired mechanisms. If the loop always runs end-to-end, an unwired component cannot exist.

## 5. Step zero before writing ANY mechanism: does it already exist?
grep the code + read `STATUS.md`; extend the owning module, never build a parallel copy. (Three near-reinventions in one
session — `_moved_body`, `ObjectOperator`, a hand-rolled forward model — all came from skipping this.)

## 6. Tests are monotonic.
The suite must not shrink silently. Removing or skipping a test requires stating so and why in the commit message.

---

**The test of whether we're obeying these:** at any moment, `STATUS.md` + the reachability test answer *"what is in the
codebase and what is wired?"* in seconds, and the answer matches reality. If it doesn't, stop and fix that before anything
else. That failure — not knowing what's going on — is the one we are here to prevent.
