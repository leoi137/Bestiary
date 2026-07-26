---
name: robotics-research
description: Investigate a robotics, RL, or simulation question against primary sources — hardware requirements, algorithms, control methods, what production stacks actually do — and land the answer as a decision with a revisit trigger or a theory note. Use before committing to any approach whose cost is measured in days. Never answers from memory.
---

# robotics-research

Answer a question about robotics, reinforcement learning, or simulation well
enough to bet days of work on it.

## The rule that matters most

**Never answer from memory.** This field moves fast, version support matrices
change quietly, and a plausible-sounding recollection is the most expensive
kind of wrong here — it survives review precisely because it sounds right.

Every load-bearing claim gets a source and a date checked. If a number decides
something, fetch the page that states it.

Worked example of the standard: the Isaac Lab question in
`research/decisions/0001-defer-isaac-lab.md`. The whole decision turned on
"minimum 16 GB VRAM, minimum RTX 4080" — numbers recalled from memory would
have been stale by a major version.

## Scope the question first

Write the question in one sentence, and say **what decision depends on it**.
A question that changes nothing does not need this skill.

Then decide what would actually settle it:

| Question type | What settles it |
|---|---|
| Will this run on our hardware? | The vendor's requirements page, plus our actual specs from `nvidia-smi` / `free -h` |
| What do production stacks do? | The source of `legged_gym`, `unitree_rl_gym`, Isaac Lab — read the code, not the blog post |
| Does this method work? | The paper's *experiments* section and its hyperparameters, not its abstract |
| How does our own code behave? | Run it. A measurement beats any citation about our own system |

## Source hierarchy

1. **Our own measurements.** Anything about this repo is settled by running
   it, never by reasoning about it.
2. **Primary vendor/project documentation**, version-pinned. Note the version
   — "Isaac Sim 5.1 requirements" is a claim; "Isaac Sim requirements" is not.
3. **Source code** of the reference implementation.
4. **Papers**, read for their experimental setup.
5. **Forum posts and blogs** — useful for *finding* the primary source and for
   knowing what breaks in practice. Never the last word on a number.

Where sources disagree, say so rather than picking the convenient one. A
documented minimum and a working community report that contradicts it are both
facts, and the gap between them is usually the interesting part.

## Check it against our constraints

Every answer gets tested against the three that actually bind here:

- **One RTX 2080, 8 GB VRAM.** Most published results assume far more. A
  method that needs 4096 parallel environments is not available to us at the
  same scale, and saying so is part of the answer.
- **A single machine, hours per experiment.** Anything requiring dozens of
  runs to tune is effectively out of reach until throughput improves.
- **We have a regression oracle we cannot afford to lose** —
  `robots/hound/check.py`. Any change that invalidates it costs more than it
  looks.

## Land the result

Research that stays in a conversation is lost. It goes to one of:

- **`research/decisions/NNNN-*.md`** — if it settles a choice. Must include
  what was verified (with sources and the date), **the trigger that would
  reverse it**, and what we gave up. Follow the shape in that folder's README.
- **`docs/theory/*.md`** — if it explains a mechanism we now depend on. Write
  it when the idea becomes load-bearing, with the math standard from
  `research/learnings/README.md`: equation, symbols with units, worked with
  our real numbers.
- **An episode's Open questions** — if it narrowed the question but did not
  settle it. Say what would settle it next.

Never leave it only in chat.

## Use a subagent for the sweep

Searching floods context with material you will not keep. Delegate the search;
keep the judgement.

Ask the subagent for: the claim, the source URL, the version it applies to,
and the date checked. Not the page contents.

For a question with several independent angles — hardware, method, what the
reference implementation does — run them as parallel subagents and synthesize
yourself.

## Report honestly

- Separate **verified** from **inferred** from **assumed**. Mark which is
  which in the write-up.
- Give a confidence, and say what would change it.
- If the honest answer is "this is uncertain and here is the cheapest
  experiment that would settle it", that is a complete answer. Propose the
  timeboxed spike rather than guessing — an afternoon spent finding out beats
  a week spent building on a recollection.
