# open-robotics-plugin(中文)

把机器人 agent 技能(MuJoCo 仿真、感知、运动规划、sim2real、真机 bring-up)一次安装到多种 agent 工具里:Claude Code、Antigravity、Codex、Gemini CLI、Cursor、Windsurf、Cline、Copilot、任意读 AGENTS.md 的工具,以及 ChatGPT(仅限仿真和代码,手动上传)。

技能内容来自 Google DeepMind 的 robotics-antigravity-plugin(Apache-2.0),详见 NOTICE。本项目与 Google 无关联。

```bash
python3 tools/orp.py list
python3 tools/orp.py install --target claude --scope user --dry-run
python3 tools/orp.py install --target claude --scope user
```

注意:
- 真机技能会移动机械臂。安装会同时带上安全规则;务必人在现场并能随时断电。
- ChatGPT 访问不到 USB/串口,不能控制真机。
- Codex/Gemini 的目标路径是尽力而为,未逐版本验证,可用 `--dest` 覆盖。
- MuJoCo Studio sidecar 只适用于 Antigravity;其他工具直接运行 `studiod.py`。
- 调用 Gemini 感知的技能需要 `GEMINI_API_KEY`。
