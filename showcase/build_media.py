"""Compose labelled clips and a montage from frames written by render_clip.py.

Usage: python build_media.py <frames_root> <out_dir>
Needs Pillow and ffmpeg on PATH. Outputs GIF (for README embedding) and MP4.
"""
import pathlib, subprocess, sys, tempfile, shutil
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
FONT = "/System/Library/Fonts/Helvetica.ttc"
def font(sz, bold=False):
    try: return ImageFont.truetype(FONT, sz, index=1 if bold else 0)
    except Exception: return ImageFont.load_default()

STEP = 2          # keep every 2nd rendered frame (about 2.4x real time at 12 fps)
FPS = 12
GREEN, RED, INK, BG, MUTED = (52, 199, 120), (235, 76, 76), (240, 244, 250), (11, 18, 32), (140, 154, 178)

def frames(name):
    fs = sorted((ROOT / name).glob("*.png"))[::STEP]
    if not fs: raise SystemExit(f"no frames for {name}")
    return fs

def panel(img, w, h, label, ok, t_frac, show_badge):
    im = img.convert("RGB").resize((w, h), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    f = font(max(11, h // 15), True)
    tw = d.textlength(label, font=f)
    d.rounded_rectangle((8, 8, 8 + tw + 16, 8 + f.size + 10), 6, fill=(11, 18, 32, 190))
    d.text((16, 12), label, font=f, fill=INK)
    if show_badge:
        txt = "SUCCESS" if ok else "FAILED"
        bf = font(max(11, h // 15), True); bw = d.textlength(txt, font=bf)
        col = GREEN if ok else RED
        d.rounded_rectangle((w - bw - 28, h - f.size - 22, w - 8, h - 8), 6, fill=col + (235,))
        d.text((w - bw - 18, h - f.size - 17), txt, font=bf, fill=(8, 14, 24))
    d.rectangle((0, h - 4, int(w * t_frac), h), fill=(255, 255, 255, 170))
    return im

def compose(specs, cols, pw, ph, title, footer, name):
    """specs: [(frames_name, label, ok)] -> encode name.gif / name.mp4"""
    seqs = [(frames(n), l, ok) for n, l, ok in specs]
    n = max(len(s[0]) for s in seqs) + 10        # hold the last frame ~0.8 s
    rows = (len(seqs) + cols - 1) // cols
    HEAD, FOOT = 46, 26
    W, H = cols * pw, rows * ph + HEAD + FOOT
    tmp = pathlib.Path(tempfile.mkdtemp())
    for i in range(n):
        canvas = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(canvas)
        d.text((14, 10), title, font=font(22, True), fill=INK)
        d.text((14, H - FOOT + 5), footer, font=font(13), fill=MUTED)
        for k, (fs, label, ok) in enumerate(seqs):
            j = min(i, len(fs) - 1)
            canvas.paste(panel(Image.open(fs[j]), pw, ph, label, ok, j / max(1, len(fs) - 1), i >= len(fs) - 1),
                         ((k % cols) * pw, HEAD + (k // cols) * ph))
        canvas.save(tmp / f"{i:04d}.png")
    gif, mp4 = OUT / f"{name}.gif", OUT / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", str(tmp / "%04d.png"),
                    "-vf", "split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    "-loop", "0", str(gif)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", str(tmp / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart", str(mp4)], check=True)
    shutil.rmtree(tmp)
    print(f"{name}: {n} frames  gif={gif.stat().st_size/1e6:.1f} MB  mp4={mp4.stat().st_size/1e6:.1f} MB")

FOOT = "MuJoCo simulation only  |  packaged v1 controller (sim2real-gauntlet)  |  tier T0, seed 0  |  ~2.4x speed"
compose([("yam_cube_in_bowl", "YAM  cube_in_bowl", True), ("yam_peg_in_plate", "YAM  peg_in_plate", True),
         ("yam_two_block_stack", "YAM  two_block_stack", True), ("so101_cube_in_bowl", "SO-101  cube_in_bowl", True),
         ("so101_peg_in_plate", "SO-101  peg_in_plate", True), ("so101_two_block_stack_fixed", "SO-101  two_block_stack (mask fix)", True)],
        3, 320, 180, "One controller, two arms, three tasks", FOOT, "hero")
compose([("so101_two_block_stack_shipped", "as shipped", False), ("so101_two_block_stack_fixed", "orange mask relaxed", True)],
        2, 480, 270, "SO-101 two_block_stack: why it failed, and the fix", FOOT, "so101_two_block_stack_before_after")
for robot, label in (("yam", "YAM"), ("so101", "SO-101")):
    compose([(f"{robot}_{t}", f"{label}  {t}", True) for t in ("cube_in_bowl", "peg_in_plate", "two_block_stack")
             if not (robot == "so101" and t == "two_block_stack")] +
            ([("so101_two_block_stack_fixed", "SO-101  two_block_stack (mask fix)", True)] if robot == "so101" else []),
            3 if robot == "yam" else 2, 480 if robot == "yam" else 480, 270, f"{label}: simulated pick-and-place", FOOT, f"{robot}_tasks")
