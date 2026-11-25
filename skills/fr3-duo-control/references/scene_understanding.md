<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

# Scene Understanding (`fr3_vqa.py`)

Use [`fr3_vqa.py`](../scripts/fr3_vqa.py) to query Gemini over
`Fr3DuoRobot.observe_scene()` frames (`head_camera_left`, `head_camera_right`,
`wrist_camera_left`, `wrist_camera_right`) and robot state when onboard sensing
alone is insufficient.

## 1. Closed-Set Decisions (`decide_scene`)

Use `decide_scene(obs, questions, cameras=..., resolution="medium")`
(`gemini-decision-1`, zero output tokens) for calibrated branching and
verification gates. `questions` is forwarded directly to
`client.decisions.create(...)` (see
https://aistudio.google.com/docs/decisions-api):

- **`noul` (yes/no):** Returns calibrated `P(true)` (`noul` in `[0.01, 0.99]`)
  and `confidence`. Add `criteria` (`true_description`, `false_description`) for
  ambiguous conditions.
- **`choice` (single-select):** Picks among 2–255 options (`criteria` map of
  option ID $\to$ description); returns `choice`, `probabilities`, `confidence`.
- **`score` (rubric):** Grades ordered discrete levels (`criteria` list);
  returns probability-weighted `score`, `probabilities`, `legend`, `confidence`.

```python
from fr3_vqa import decide_scene

res = decide_scene(
    obs,
    questions={
        "condition_met": {
            "type": "noul",
            "instructions": "Did the scene reach the expected post-condition?",
        },
        "next_branch": {
            "type": "choice",
            "instructions": "Which branch should execute next?",
            "criteria": {
                "proceed": "Target state verified; continue.",
                "retry": "Misaligned or incomplete; retry phase.",
                "abort": "Unexpected obstacle or unsafe state.",
            },
        },
    },
    cameras=("head_camera_left", "wrist_camera_left"),
    resolution="medium",  # "low" | "medium" | "high" | "ultra_high"
)
if res["status"] == "ok":
  ans = res["answers"]["condition_met"]
  passed = ans["noul"] >= 0.8 and ans["confidence"] >= 0.7
```

Use `confidence` to distinguish clear outcomes from uncertain ones (retry or
escalate to `ask_scene_vqa` / operator review when low).

## 2. Open-Ended VQA (`ask_scene_vqa`)

Use `ask_scene_vqa(obs, question, cameras=..., config=...)`
(`gemini-4-hydrogen` default; optional `config` forwarded to
`client.models.generate_content(...)`) for open-ended reasoning, scene
inventory, or diagnosing unexpected failures.

## 3. Offline VLM Fallback

If `GEMINI_API_KEY` is unset or `google-genai` is missing, both functions write
camera PNGs, `contact_sheet_2x2.png`, `observation.json`, and
`decision_request.json` / `vqa_request.json` to `output_dir` (default: stage VQA
logs dir) and return `status == "fallback_saved"` for direct `read_file`
inspection.
