"""Draw the README banner (needs Pillow). Usage: python make_banner.py <out.png>"""
import sys
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 420
FONT = "/System/Library/Fonts/Helvetica.ttc"
def f(sz, bold=False):
    return ImageFont.truetype(FONT, sz, index=1 if bold else 0)

img = Image.new("RGB", (W, H))
px = img.load()
for y in range(H):                      # vertical gradient
    for x in range(W):
        t = x / W * 0.5 + y / H * 0.5
        px[x, y] = (int(8 + 22 * t), int(14 + 26 * t), int(30 + 60 * t))
d = ImageDraw.Draw(img, "RGBA")
for gx in range(0, W, 40):              # faint grid
    d.line((gx, 0, gx, H), fill=(120, 160, 255, 14))
for gy in range(0, H, 40):
    d.line((0, gy, W, gy), fill=(120, 160, 255, 14))
d.ellipse((W - 520, -220, W + 120, 420), fill=(70, 130, 255, 38))   # glow
d.ellipse((W - 380, 120, W + 60, 560), fill=(255, 120, 190, 26))

d.text((80, 70), "open-robotics-plugin", font=f(88, True), fill=(244, 247, 255))
d.text((84, 178), "One robotics skill pack. Every coding agent.", font=f(40), fill=(176, 196, 232))
d.text((84, 236), "MuJoCo sim  /  perception  /  motion planning  /  sim-to-real  /  YAM, SO-101, FR3", font=f(24), fill=(120, 140, 178))

chips = ["Claude Code", "Antigravity", "Codex", "Gemini CLI", "Cursor", "Windsurf", "Cline", "Copilot", "AGENTS.md", "ChatGPT"]
x, y = 84, 306
for c in chips:
    w = int(d.textlength(c, font=f(22, True))) + 36
    if x + w > W - 60:
        x, y = 84, y + 52
    d.rounded_rectangle((x, y, x + w, y + 40), 20, fill=(255, 255, 255, 22), outline=(160, 190, 255, 90), width=1)
    d.text((x + 18, y + 7), c, font=f(22, True), fill=(226, 234, 252))
    x += w + 12
img.save(sys.argv[1], optimize=True)
