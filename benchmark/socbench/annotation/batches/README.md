# Annotation batches

Each directory is one batch: what is labelled (case ids, tasks and a digest in `batch.json`), and the reviewers' labels, one file per reviewer. Case text never enters the repository; each batch's README says how to regenerate it from pinned public sources.

| Purpose | Reviewers | Tier when complete | Ledger label |
|---|---|---|---|
| `pilot` | exactly one | B | SINGLE-REVIEW RESEARCH |
| `gold` | two independent, plus a third-person adjudicator for disagreements | A | TIER-A GOLD |

Create a batch with `jef-bench annotate new-batch`. CI runs `jef-bench annotate check-batch --git` on every pull request that touches a batch, and refuses one that:

- carries any field beyond the label fields (no state, evidence text or model output);
- labels a case outside the batch, or with a verdict the task does not allow;
- has a reviewer's file committed by more than one author, both reviewers' files committed by one author, or one commit touching both;
- has a decision by anyone but the batch's adjudicator, on anything but a disagreement, or committed by a reviewer's author.

Git authorship is what the committer configured, so these checks catch shortcuts and mistakes, not a determined impostor. The labelbook's rule stands: reviewer B is a different qualified person, never a second handle and never an LLM.
