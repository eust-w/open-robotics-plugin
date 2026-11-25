---
name: iteration
description: >-
  Staged execution and video-based debugging workflow for physical or MuJoCo
  simulated robots: splitting a robot manipulation task into stages, testing
  each stage with one script, reviewing rollout videos, and verifying physical
  completion after release and settle. Use only when writing code that makes a
  physical or simulated robot act and move. Don't use for non-robotics software
  engineering, UI/WebGL testing, or general scripting.
metadata:
  publisher: google
---

# Iterating on robot code

## Work in stages
- Split a long task into stages, one per sub-goal. For example, putting five
  objects in a box is five stages. An action that already works reliably,
  such as a closed-loop pick-and-place, counts as one stage.
- Write each stage as one script that observes, plans, acts, measures the
  result, and prints it. Run it with a timeout, for example
  `timeout 300 python3 stage_1.py`.
- After each stage, use the measured result to decide whether to move on. If
  the stage failed, look at the trajectory to find where it went wrong and fix
  the code. Then measure the scene again and go back to the first stage whose
  goal no longer holds.
- Do not put multi-line code in `python3 -c`. Write it to a file instead.

## Get a second opinion when stuck
Use this when you can record video and start subagents.
- When a stage fails repeatedly, give a separate subagent the video of the
  last try (a video, not still images) and ask it to describe what happened
  and what went wrong. Send only the video, not your code or what you
  expected, so that its description is not biased.
- If the stage still fails after that, give a separate subagent the videos,
  your code, and the measured results, and ask what to change.

## Done means measured
- Decide success from measurements taken after the robot has let go and the
  scene has settled, not from commanded poses or a printed "SUCCESS". Check
  every goal condition before you finish.
