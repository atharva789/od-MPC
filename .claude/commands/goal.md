---
description: Frame a work session against od-MPC's current milestone before doing anything
---

# /goal

Every session on this repository starts here. The point is to work on the thing
that actually unblocks progress, not the thing that is easiest to start.

Do not skip to editing code. Work the four steps in order.

## 1. State the goal

Read [`docs/SPEC.md`](../../docs/SPEC.md) and identify:

- **The current milestone** — the first row in §5 whose exit criterion is not met.
- **Its exit criterion** — the literal, checkable condition. Not "improve the
  collector"; the actual sentence in the table.
- **What blocks it** — cross-reference §3 (Assumptions) and §7 (Open questions).

State all three explicitly before continuing. If the milestone is unclear or the
exit criterion is not checkable, fixing the spec *is* the goal for this session.

## 2. Assess honestly

Establish where things actually stand, from evidence rather than memory:

```bash
pytest -q                       # must be green before any new work
git log --oneline -8            # what recent sessions actually landed
```

Compare against the roadmap in `README.md`. If a checkbox claims something the
tests or the code do not support, correct the claim. **An inaccurate status
line is a bug** — this repo is read by people evaluating whether the work is
real, and overstating it is worse than an empty checkbox.

## 3. Pick exactly one action

Choose the single highest-leverage move toward the current exit criterion.
Prefer, in order:

1. **Unblock an open question.** Resolving a §7 item beats writing new code that
   might be built on a wrong assumption. Assumption **A1** (the loader's record
   key for continuous actions) blocks milestone M2 and therefore everything
   after it.
2. **Close a gap in tested logic.** Modules under `od_mpc/` that pure-Python
   tests could cover but do not.
3. **Improve accuracy of the docs.** Wrong facts in `SPEC.md` compound.

Explicitly do **not**:

- Add features beyond the current milestone.
- Touch `od_mpc/sim/adapter.py` behaviour without a real innate-os checkout to
  verify against; guessing at an API that was verified from source is a
  regression.
- Copy any code, config, or snippet from open-dreamer. It is All Rights
  Reserved. See §6 of the spec. Describing its interface is fine; reproducing
  its source is not, including in documentation.

## 4. Report against the goal

Finish with:

- The goal you stated in step 1.
- What you changed, and the evidence it works (test output, not assertion).
- Whether the exit criterion is now met. If not, what specifically remains.
- Anything you learned that belongs in `SPEC.md` §2 (verified) or §3
  (assumptions) — move facts between those sections as they get confirmed.

If you could not make progress, say so plainly and record why in §7. A session
that accurately reports being blocked is more useful than one that invents
motion.
