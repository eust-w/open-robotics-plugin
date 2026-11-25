---
name: share-example
description: >-
  Packages a real-robot or simulation demo (compact transcript.jsonl, final
  files, videos under 10 MB) into a scrubbed bundle with a pre-filled Show and
  tell GitHub Discussions URL. Use only when sharing a robotics example to
  GitHub Discussions. Don't use for creating repos, PRs, or issues.
metadata:
  publisher: google
---

# Share Robotics Example to GitHub Discussions

Packages a real-robot task or MuJoCo simulation demo into two drag-and-drop
files (`demo.mp4` and `bundle.zip`) and a pre-filled **Show and tell** URL for
`https://github.com/google-deepmind/robotics-antigravity-plugin/discussions`.
Shares only to GitHub Discussions. Never creates a GitHub repo or requires `gh`.

## Steps

1. **Gather inputs**:
   * Compact transcript:
     `<appDataDir>/brain/<conversation-id>/.system_generated/logs/transcript.jsonl`
   * Final workspace files (controller, robot/sim scripts, `scene.xml`, etc.),
     skipping `.env`, `.git`, `.venv`, `__pycache__`, `node_modules`, and
     `*.mcap`.
   * Demo videos (`.mp4`, `.mov`, or `.gif` from real robot or simulation).
   * Short title (e.g. `[Real SO-101] Pick Marker` or `[Sim] YAM Espresso`),
     prompt used, and setup (real robot model or sim-only + OS).

2. **Scrub and package into `./share_post/`**:
   * Copy the compact `transcript.jsonl` and final text files, redacting API
     keys/tokens (`AIza...`, `GEMINI_API_KEY`, `ghp_...`, `sk-...`, `Bearer`)
     and replacing local home paths (`/Users/<name>`, `/home/<name>`) with `~`.
   * Write a short `README.md` (title, setup, prompt, file list) and zip
     `README.md`, scrubbed `transcript.jsonl`, and final files into
     `./share_post/bundle.zip` (must be under 25 MB).
   * Copy demo video(s) into `./share_post/`. If an `.mp4`/`.mov` exceeds 10 MB
     and `ffmpeg` is installed, compress it to H.264 (`yuv420p`) under 10 MB.

3. **Review and approve (mandatory)**:
   Show the user the files inside `bundle.zip`, redaction counts, video sizes
   (remind them to check videos for private lab details), and the draft
   Discussion title and body. Wait for explicit approval.

4. **Post to Show and tell**:
   Build a URL-encoded pre-filled link to
   `https://github.com/google-deepmind/robotics-antigravity-plugin/discussions/new`
   with `category=show-and-tell`, `title`, and `body` (Demo, Prompt, Setup,
   Files & Conversation). Give the user the link and `./share_post/` so they
   can open it, drag the video(s) and `bundle.zip` into the editor, and click
   **Start discussion**.
