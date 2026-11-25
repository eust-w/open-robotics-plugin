# Why SO-101 `two_block_stack` fails with the packaged v1 controller

Date: 2026-11-25. Follow-up to `../sim2real-gauntlet-run/RUN_RECORD.md`, where v1
solved only 2 of 20 SO-101 `two_block_stack` episodes (YAM: 16 of 20). Pure
simulation, skill `sim2real-gauntlet` as installed by open-robotics-plugin
(upstream snapshot v0.1.13). The installed skill was **not modified**; every
experiment monkeypatches in the Python process only.

## Root cause

The controller's orange colour mask rejects most of the orange block's pixels in
the SO-101 overhead view, so it locates the block from a thin edge strip and aims
about 15 mm off-centre. The block is 30 mm wide, so the open jaw lands on its edge.

Evidence, in the order it was found (scripts and raw output are in this folder):

1. **The block is never carried.** `diagnose.py` (outputs `diag_*.txt`): in the
   failing T0 episode the top block lifts at most 3.2 mm, tilts up to 21.8 degrees
   around tick 192-216 while the gripper is still open (~97), and settles back
   near its start. The base block never moves. In the working `cube_in_bowl`
   episode the cube lifts 62 mm.
2. **The jaw closes on nothing.** Gripper reading falls to ~15 when the
   block is missed, versus ~21 when it holds the cube.
3. **The controller's estimate is wrong.** `trace_perception.py` (outputs
   `trace_*.txt`), true target x = -0.070 m:

   | | perceived target xy | error |
   |---|---|---|
   | `cube_in_bowl` (red cube) | (-0.0684, -0.0500) | 1.6 mm |
   | `two_block_stack` (orange block) | (-0.0550, -0.0498) | **15 mm** |

4. **Why.** The target mask returns 56 pixels for the red cube but only 7 for
   the orange block, all in one column at x = -0.055. All 56 block pixels pass
   the shared red-family test, but the orange mask's second clause
   `(r-g) > 0.42*max(0, g-b)` rejects 88% of them. The block's top face renders
   over-exposed: median pixel about (255, 226, 118), a pale yellow. That clause
   exists to reject yellow, so the lit top face is treated as yellow and only the
   darker edge strip survives. The centroid of that strip is the block's edge.
5. **Causal check.** `fix_experiment.py` drops only that second clause, in-process,
   and reruns the packaged v1 controller (T0-T3, 2 episodes each, seed 0):

   | Orange mask | success | safe-pass |
   |---|---|---|
   | as shipped | 1/8 | 1/8 |
   | clause removed | **7/8** | 7/8 |

   Same seeds, same controller, one changed line of the mask.

## What this does and does not show

- The failure is a perception-threshold problem at this exposure, not a
  motion-planning or stacking-logic problem: with the mask relaxed the same
  controller stacks the blocks.
- Why the YAM is fine was **not** investigated. A plausible reason is different
  camera/lighting on the YAM rig leaving the top face less clipped; I did not
  check that, so treat it as a hypothesis.
- Only the first `perceive_overhead` call was inspected closely. 8 episodes per
  arm of the experiment is small; one patched T3 episode still failed (cause not
  examined). It shows the mask is sufficient to explain most of the failure, not that
  nothing else is wrong.
- Removing the clause could let yellow-ish objects in. The only yellow object in
  these scenes is the bystander cube, which already fails the shared `r-g > 26`
  test (r-g is about 15 for its colour), but I did not test other scenes. A real fix
  would want an exposure-robust hue test, not just a deleted clause.
- Note on tooling: in-process runs in these scripts return `None` from the
  controller wrapper. Calling `cbc.run` directly and returning its tick count makes
  the harness report a `crash` outcome, an artefact of my first trace script, not a
  controller failure.

## Files

- `diagnose.py`: replays an episode, prints block poses and gripper trace over time.
- `trace_perception.py`: instruments perception and motion calls, prints what the
  controller believed next to ground truth and the colour-mask breakdown.
- `fix_experiment.py`: baseline vs relaxed-mask comparison.
- `diag_*.txt`, `trace_*.txt`, `fix_experiment_*.txt`: raw output of the runs above.

## Possible upstream report

If filed, it should say: "`color_mask('orange')` clause 2 rejects over-exposed
orange block top faces in the SO-101 overhead render; `two_block_stack` target
xy is then ~15 mm off and the grasp misses. Dropping that clause lifts a paired
T0-T3 smoke test from 1/8 to 7/8." I have not filed anything.
