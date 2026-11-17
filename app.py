#!/usr/bin/env python3
"""
Dropout: a local web UI for tracing flat-color artwork to SVG while dropping
the colors you don't want (black, shadows, backgrounds...).

Setup:
    pip install flask vtracer opencv-python numpy pillow
    pip install cairosvg        # optional, enables PDF export

Run:
    python app.py               # opens http://127.0.0.1:5757
"""
import collections
import io
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
import os

import cv2
import numpy as np
import vtracer
from flask import Flask, Response, jsonify, request
from PIL import Image

try:
    import cairosvg
    HAVE_PDF = True
except Exception:  # cairo missing is fine, PDF export is just disabled
    HAVE_PDF = False

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024

MAX_SIDE = 4096              # larger uploads are scaled down to this
MAX_TRACE_PIXELS = 70_000_000  # caps the working resolution (memory guard)
STORE = collections.OrderedDict()
LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Image + palette helpers
# --------------------------------------------------------------------------
def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb2hex(c):
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c)


def saturation(rgb):
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    return (mx - mn) / np.maximum(mx, 1)


def detect_palette(rgba, max_colors=10):
    """Find the flat colors in the artwork. Anti-aliased edge blends are spread
    over many histogram bins, so they never beat the real colors."""
    solid = rgba[..., 3] >= 128
    px = rgba[..., :3][solid]
    if len(px) == 0:
        return []
    if len(px) > 400_000:
        px = px[np.random.RandomState(0).choice(len(px), 400_000, replace=False)]

    q = (px >> 3).astype(np.int32)
    key = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    total = len(px)

    picked = []
    for o in order[:250]:
        if counts[o] < total * 0.003:
            break
        c = px[inv == o].astype(np.float32).mean(0)
        if all(np.linalg.norm(c - p) > 60 for p in picked):
            picked.append(c)
            if len(picked) >= max_colors:
                break

    P = np.array(picked, np.float32)
    nearest = np.linalg.norm(px[:, None, :].astype(np.float32) - P[None], axis=2).argmin(1)
    share = np.bincount(nearest, minlength=len(P)) / total * 100

    has_alpha = bool((rgba[..., 3] < 250).any())
    out = []
    for c, s in sorted(zip(picked, share), key=lambda t: -t[1]):
        near_black = c.max() < 110 and saturation(c) < 0.2
        near_white = c.min() > 240
        keep = not (near_black or (near_white and not has_alpha))
        out.append({"hex": rgb2hex(c), "pct": round(float(s), 1), "keep": keep})
    return out


# --------------------------------------------------------------------------
# Tracing
# --------------------------------------------------------------------------
def classify(rgba, palette, dark_cutoff):
    """Label every pixel with the index of its palette color.

    Pixels may only match palette colors with a similar saturation, so a
    cream/black edge blend becomes cream or black, never coral or red.
    """
    rgb = rgba[..., :3].astype(np.float32)
    P = np.array([hex2rgb(c["hex"]) for c in palette], np.float32)
    keep = [bool(c["keep"]) for c in palette]
    n = len(P)

    mx = rgb.max(2)
    sat = saturation(rgb)
    psat = saturation(P)

    dist = np.stack([np.linalg.norm(rgb - P[i], axis=2) for i in range(n)])
    allowed = np.abs(sat[None] - psat[:, None, None]) < 0.3
    allowed |= (mx < 60)[None]
    lab = np.where(allowed, dist, np.inf).argmin(0)
    none = ~allowed.any(0)
    if none.any():
        lab[none] = dist.argmin(0)[none]

    # Dark grey cutoff: only when a dark neutral color is set to Drop.
    dark_drop = [i for i in range(n) if not keep[i] and psat[i] < 0.2 and P[i].max() < 100]
    if dark_drop and dark_cutoff > 0:
        lab[(sat < 0.35) & (mx < dark_cutoff)] = dark_drop[0]
    return lab, keep


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smooth_contour_points(cnt, passes, alpha):
  """Average points along a closed contour while preserving sharp anchors."""
  if cnt is None or len(cnt) < 8:
    return cnt
  pts = cnt[:, 0, :].astype(np.float32)
  for _ in range(passes):
    prev2 = np.roll(pts, 2, axis=0)
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    nxt2 = np.roll(pts, -2, axis=0)

    # 5-point moving average along the curve.
    avg = (prev2 + 2 * prev + 3 * pts + 2 * nxt + nxt2) / 9.0

    # Preserve sharp corners so important anchors survive heavy polish.
    v1 = pts - prev
    v2 = nxt - pts
    n1 = np.linalg.norm(v1, axis=1) + 1e-6
    n2 = np.linalg.norm(v2, axis=1) + 1e-6
    cosang = np.clip((v1 * v2).sum(1) / (n1 * n2), -1.0, 1.0)
    corner_strength = np.clip((1.0 - cosang) * 0.5, 0.0, 1.0)
    local_alpha = alpha * (1.0 - 0.82 * corner_strength)
    pts = pts * (1.0 - local_alpha[:, None]) + avg * local_alpha[:, None]

  out = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
  return out


def contour_average_smooth_mask(on, polish):
  """Smooth mask boundaries by averaging along contour points."""
  if polish <= 0:
    return on
  m = (on.astype(np.uint8) * 255)
  contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
  if not contours or hierarchy is None:
    return on

  passes = 1 + int(round(polish * 5))
  alpha = 0.22 + 0.5 * polish
  hierarchy = hierarchy[0]
  out = np.zeros_like(m)

  for i, c in enumerate(contours):
    if len(c) < 8 or abs(cv2.contourArea(c)) < 6:
      smoothed = c
    else:
      smoothed = smooth_contour_points(c, passes, alpha)
    is_hole = hierarchy[i][3] != -1
    color = 0 if is_hole else 255
    cv2.drawContours(out, [smoothed], -1, color, thickness=cv2.FILLED)

  return out > 127


def trace_mask(mask, scale, sigma, thresh, corner, speckle, trap, polish,
               length_threshold, splice_threshold, max_iterations, path_precision,
               tmpdir, name):
    # Linear interpolation keeps contours smooth without the stair-step look
    # that nearest-neighbor introduces on diagonals.
    big = cv2.resize(mask.astype(np.uint8) * 255, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_LINEAR)
    if sigma > 0:
        big = cv2.GaussianBlur(big, (0, 0), sigma)
    elif scale > 1:
        # Mild blur at upscaled sizes reduces quantization jaggies while
        # preserving shape fidelity.
        big = cv2.GaussianBlur(big, (0, 0), 0.35 * scale)
    on = big > int(thresh * 255)
    if trap > 0:
        r = max(1, int(round(trap * scale)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        on = cv2.dilate(on.astype(np.uint8), k) > 0

    # Optional post-process polish pass (after core tracing settings):
    # average points along each contour for Illustrator-like curve smoothing.
    on = contour_average_smooth_mask(on, polish)

    binary = np.where(on, 0, 255).astype(np.uint8)   # VTracer traces dark pixels

    png = os.path.join(tmpdir, name + ".png")
    svg = os.path.join(tmpdir, name + ".svg")
    cv2.imwrite(png, binary)

    # Run vtracer in a child process so native crashes don't take down Flask.
    child_code = (
        "import vtracer\n"
        "vtracer.convert_image_to_svg_py(\n"
        "    r'''{png}''', r'''{svg}''', colormode='binary', mode='spline',\n"
        "    filter_speckle={speckle}, corner_threshold={corner},\n"
        "    length_threshold={length_threshold}, splice_threshold={splice_threshold},\n"
        "    path_precision={path_precision}, max_iterations={max_iterations},\n"
        ")\n"
    ).format(
        png=png,
        svg=svg,
        speckle=max(1, int(round(speckle * scale))),
        corner=int(corner),
        length_threshold=round(length_threshold, 2),
        splice_threshold=int(round(splice_threshold)),
        path_precision=int(path_precision),
        max_iterations=int(max_iterations),
    )
    proc = subprocess.run(
        [sys.executable, "-c", child_code],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "vtracer failed (code %s)%s"
            % (
                proc.returncode,
                ": " + proc.stderr.strip() if proc.stderr.strip() else "",
            )
        )

    with open(svg) as f:
        text = f.read()
    return re.findall(r'<path d="([^"]+)"[^>]*?transform="([^"]+)"', text)


def run_trace(rgba, palette, p):
    h, w = rgba.shape[:2]
    scale = int(p.get("scale", 2))
    scale = max(1, min(scale, int((MAX_TRACE_PIXELS / (w * h)) ** 0.5)))

    lab, keep = classify(rgba, palette, float(p.get("dark", 110)))
    solid = rgba[..., 3] >= float(p.get("alpha", 128))

    base_sigma = clamp(float(p.get("sigma", 0.125)), 0.0, 0.8) * scale
    thresh = clamp(float(p.get("thresh", 0.56)), 0.05, 0.95)
    corner = clamp(float(p.get("corner", 28)), 1.0, 100.0)
    speckle = clamp(float(p.get("speckle", 0.25)), 0.0, 20.0)
    trap = clamp(float(p.get("trap", 0)), 0.0, 10.0)
    polish = clamp(float(p.get("polish", 0.0)), 0.0, 1.0)

    # Higher simplify should smooth long curves without deleting important
    # anchors, so keep pruning mild and drive smoothing mostly through sigma.
    simplify = clamp(float(p.get("simplify", 0.35)), 0.0, 1.0)
    sigma = base_sigma + (0.18 * simplify * scale)
    sigma = clamp(sigma, 0.0, 1.4 * scale)

    # Keep these ranges conservative to avoid wiping small anchors.
    length_threshold = 2.0 + simplify * 5.0
    splice_threshold = 35.0 + simplify * 65.0
    max_iterations = int(round(24 + simplify * 4))
    path_precision = 3

    # Counterbalance simplification so corner-heavy shapes keep key points.
    corner = clamp(corner - simplify * 6.0, 1.0, 100.0)

    kept = [i for i, k in enumerate(keep) if k]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">'
             % (w * scale, h * scale, w, h)]
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for order, i in enumerate(kept):
            mask = (lab == i) & solid
            if not mask.any():
                continue
            top = order == len(kept) - 1
            paths = trace_mask(mask, scale, sigma, thresh, corner, speckle,
                               0 if top else trap, polish,
                               length_threshold, splice_threshold,
                               max_iterations, path_precision,
                               tmp, "layer%d" % i)
            total += len(paths)
            parts.append('<g id="color-%s" fill="%s">' % (palette[i]["hex"].lstrip("#"), palette[i]["hex"]))
            parts += ['<path d="%s" transform="%s"/>' % (d, t) for d, t in paths]
            parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts), total


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
def remember(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, "PNG")
    key = uuid.uuid4().hex
    with LOCK:
        STORE[key] = (rgba, buf.getvalue())
        while len(STORE) > 6:
            STORE.popitem(last=False)
    return key


@app.post("/api/analyze")
def analyze():
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file received."), 400
    try:
        im = Image.open(f.stream)
        im = im.convert("RGBA")
    except Exception:
        return jsonify(error="Couldn’t read that file. Use a PNG, JPG or WebP."), 400
    if max(im.size) > MAX_SIDE:
        r = MAX_SIDE / max(im.size)
        im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
    rgba = np.array(im)
    palette = detect_palette(rgba)
    if not palette:
        return jsonify(error="That image has no visible pixels."), 400
    return jsonify(id=remember(rgba), w=im.width, h=im.height, palette=palette, pdf=HAVE_PDF)


@app.get("/api/palette/<key>")
def palette_again(key):
    with LOCK:
        item = STORE.get(key)
    if not item:
        return jsonify(error="That image expired. Load it again."), 410
    return jsonify(palette=detect_palette(item[0]))


@app.get("/api/image/<key>")
def image(key):
    with LOCK:
        item = STORE.get(key)
    if not item:
        return "Not found", 404
    return Response(item[1], mimetype="image/png")


@app.post("/api/trace")
def trace():
    data = request.get_json(force=True)
    with LOCK:
        item = STORE.get(data.get("id"))
        if item:
            STORE.move_to_end(data.get("id"))
    if not item:
        return jsonify(error="That image expired. Load it again."), 410
    t0 = time.time()
    try:
        svg, n = run_trace(item[0], data["palette"], data.get("params", {}))
    except Exception as e:  # surface the reason instead of a blank preview
        return jsonify(error="Tracing failed: %s" % e), 500
    return Response(svg, mimetype="image/svg+xml",
                    headers={"X-Paths": str(n), "X-Seconds": "%.1f" % (time.time() - t0)})


@app.post("/api/pdf")
def pdf():
    if not HAVE_PDF:
        return jsonify(error="Install cairosvg to export PDF."), 501
    out = cairosvg.svg2pdf(bytestring=request.get_data())
    return Response(out, mimetype="application/pdf")


@app.get("/")
def index():
    return Response(INDEX, mimetype="text/html")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
INDEX = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dropout</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#E6E8EC; --panel:#fff; --ink:#14161A; --muted:#667085; --line:#D9DCE2;
  --soft:#F3F4F6; --accent:#0B5CFF; --danger:#C2261D;
  --font:"Schibsted Grotesk",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 var(--font);font-variant-numeric:tabular-nums}
button{font:inherit;color:inherit;cursor:pointer}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

header{height:52px;display:flex;align-items:center;gap:14px;padding:0 18px;background:var(--panel);border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:9px;font-weight:700;font-size:16px;letter-spacing:-.01em}
.brand svg{width:22px;height:22px}
.fname{color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:40vw}
.spacer{flex:1}
.btn{border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:7px 13px;font-weight:500}
.btn:hover{background:var(--soft)}
.btn.primary{background:var(--ink);color:#fff;border-color:var(--ink)}
.btn.primary:hover{background:#2a2e36}
.btn[disabled]{opacity:.45;cursor:not-allowed}

/* empty state */
#empty{height:calc(100% - 52px);display:grid;place-items:center;padding:24px}
.drop{position:relative;width:min(620px,100%);padding:68px 40px;text-align:center;cursor:pointer}
.drop h1{margin:0 0 10px;font-size:34px;line-height:1.1;letter-spacing:-.025em}
.drop p{margin:0 auto 26px;max-width:440px;color:var(--muted)}
.drop .hint{margin:18px 0 0;font-size:13px}
.drop.over{background:rgba(11,92,255,.06)}
.mark{position:absolute;width:26px;height:26px;color:var(--ink)}
.mark.tl{left:0;top:0}.mark.tr{right:0;top:0;transform:scaleX(-1)}
.mark.bl{left:0;bottom:0;transform:scaleY(-1)}.mark.br{right:0;bottom:0;transform:scale(-1)}

/* workspace */
#app{display:none;height:calc(100% - 52px);grid-template-columns:minmax(0,1fr) 348px}
#app.on{display:grid}
#stage{position:relative;overflow:hidden;touch-action:none;user-select:none;cursor:grab;
  background-color:#fff;
  background-image:linear-gradient(45deg,#e9eaee 25%,transparent 25%,transparent 75%,#e9eaee 75%),
                   linear-gradient(45deg,#e9eaee 25%,transparent 25%,transparent 75%,#e9eaee 75%);
  background-size:20px 20px;background-position:0 0,10px 10px}
#stage[data-bg="white"]{background-image:none;background-color:#fff}
#stage[data-bg="dark"]{background-image:none;background-color:#1b1d22}
#stage.dragging{cursor:grabbing}
#stage.pick{cursor:crosshair}
.layer{position:absolute;inset:0;pointer-events:none}
.world{position:absolute;left:0;top:0;transform-origin:0 0}
.world img{display:block;max-width:none;image-rendering:auto}
#divider{position:absolute;top:0;bottom:0;width:0;border-left:1.5px solid var(--ink);cursor:ew-resize;touch-action:none}
#stage[data-bg="dark"] #divider{border-left-color:#fff}
#divider .grip{position:absolute;top:50%;left:0;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;
  background:var(--panel);border:1.5px solid var(--ink);display:grid;place-items:center;box-shadow:0 1px 6px rgba(0,0,0,.18)}
#divider .hit{position:absolute;top:0;bottom:0;left:-12px;width:24px}
[data-view="orig"] #divider,[data-view="vec"] #divider{display:none}

.float{position:absolute;display:flex;gap:6px;align-items:center;background:var(--panel);border:1px solid var(--line);
  border-radius:10px;padding:4px;box-shadow:0 2px 10px rgba(20,22,26,.08);cursor:default}
#viewbar{left:14px;top:14px}
#zoombar{left:14px;bottom:14px}
#statusbar{right:14px;bottom:14px;padding:6px 12px;color:var(--muted);gap:9px}
.seg{display:flex;gap:2px;background:var(--soft);border-radius:7px;padding:2px}
.seg button{border:0;background:transparent;border-radius:5px;padding:5px 10px;font-weight:500;color:var(--muted)}
.seg button[aria-pressed="true"]{background:var(--panel);color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.14)}
.ico{border:0;background:transparent;border-radius:6px;width:30px;height:30px;font-size:17px;line-height:1}
.ico:hover{background:var(--soft)}
.zlabel{min-width:48px;text-align:center;color:var(--muted);font-size:13px}
.bgsw{width:22px;height:22px;border-radius:50%;border:1.5px solid var(--line);padding:0}
.bgsw[aria-pressed="true"]{border-color:var(--accent);box-shadow:0 0 0 2px #fff,0 0 0 3.5px var(--accent)}
.spin{width:14px;height:14px;border-radius:50%;border:2px solid var(--line);border-top-color:var(--accent);animation:sp .8s linear infinite;display:none}
.busy .spin{display:block}
@keyframes sp{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion:reduce){.spin{animation-duration:2.4s}}
.err{color:var(--danger)}

/* sidebar */
aside{background:var(--panel);border-left:1px solid var(--line);overflow:auto;display:flex;flex-direction:column}
.sec{padding:16px 18px;border-bottom:1px solid var(--line)}
.sec h2{margin:0 0 12px;font-size:14px;font-weight:700;display:flex;align-items:center;justify-content:space-between}
.sec h2 .link{font-weight:500;font-size:13px;color:var(--accent);background:none;border:0;padding:0}
.sec h2 .link:hover{text-decoration:underline}
.note{margin:10px 0 0;color:var(--muted);font-size:13px}

.sw{display:grid;grid-template-columns:30px 1fr auto auto;gap:10px;align-items:center;padding:6px 0}
.chip{width:30px;height:30px;border-radius:7px;border:1px solid rgba(0,0,0,.16)}
.sw.dropped .chip{opacity:.4;background-image:linear-gradient(135deg,transparent 46%,rgba(0,0,0,.55) 46%,rgba(0,0,0,.55) 54%,transparent 54%)}
.sw .hex{font-weight:600}
.sw .pct{color:var(--muted);font-size:12.5px;display:block;font-weight:400}
.sw .x{border:0;background:none;color:var(--muted);width:22px;height:22px;border-radius:5px;font-size:15px}
.sw .x:hover{background:var(--soft);color:var(--ink)}
.sw .seg button{padding:4px 9px;font-size:13px}
.sw .seg button.on-keep[aria-pressed="true"]{background:var(--ink);color:#fff}
.sw .seg button.on-drop[aria-pressed="true"]{background:#fff;color:var(--danger)}

.ctl{margin:0 0 15px}
.ctl:last-child{margin-bottom:0}
.ctl .row{display:flex;justify-content:space-between;margin-bottom:5px}
.ctl .val{color:var(--muted)}
.ctl .ends{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-top:2px}
input[type=range]{width:100%;margin:0;accent-color:var(--accent)}

.quality{display:flex;flex-direction:column;gap:10px}
.quality .seg{width:100%}.quality .seg button{flex:1}
.dl{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.dl .btn{padding:9px 6px;text-align:center}
.dl .big{grid-column:1/-1}
.pickbtn{width:100%;margin-top:10px;display:flex;align-items:center;justify-content:center;gap:8px}
.pickbtn[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);background:rgba(11,92,255,.06)}

@media (max-width:860px){
  #app.on{display:block;height:auto}
  #stage{height:62vh}
  aside{border-left:0;border-top:1px solid var(--line)}
}
</style>
</head>
<body>
<header>
  <div class="brand">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><circle cx="12" cy="12" r="6.5"/><path d="M12 1.5v21M1.5 12h21"/></svg>
    Dropout
  </div>
  <span class="fname" id="fname"></span>
  <span class="spacer"></span>
  <button class="btn" id="newBtn" hidden>New image</button>
</header>

<section id="empty">
  <div class="drop" id="drop" tabindex="0" role="button" aria-label="Choose an image to trace">
    <svg class="mark tl" viewBox="0 0 26 26" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 25V1h24M1 13h8M13 1v8"/></svg>
    <svg class="mark tr" viewBox="0 0 26 26" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 25V1h24M1 13h8M13 1v8"/></svg>
    <svg class="mark bl" viewBox="0 0 26 26" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 25V1h24M1 13h8M13 1v8"/></svg>
    <svg class="mark br" viewBox="0 0 26 26" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 25V1h24M1 13h8M13 1v8"/></svg>
    <h1>Trace an image, drop the colors you don’t want</h1>
    <p>Pick which colors stay and which disappear, then export clean vector shapes as SVG, PNG or PDF.</p>
    <button class="btn primary" id="chooseBtn">Choose an image</button>
    <p class="hint">or drop a file here, or paste with Ctrl+V</p>
  </div>
  <input type="file" id="file" accept="image/png,image/jpeg,image/webp,image/gif,image/bmp" hidden>
</section>

<main id="app">
  <div id="stage" data-view="split" data-bg="checker">
    <div class="layer" id="layerOrig"><div class="world"><img id="imgOrig" alt="Original image"></div></div>
    <div class="layer" id="layerVec"><div class="world"><img id="imgVec" alt="Traced vector"></div></div>
    <div id="divider" role="slider" tabindex="0" aria-label="Compare original and vector" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50">
      <div class="hit"></div>
      <div class="grip"><svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="9" cy="9" r="4.5"/><path d="M9 1v16M1 9h16"/></svg></div>
    </div>

    <div class="float" id="viewbar">
      <div class="seg" id="viewSeg">
        <button data-view="orig" aria-pressed="false">Original</button>
        <button data-view="split" aria-pressed="true">Compare</button>
        <button data-view="vec" aria-pressed="false">Vector</button>
      </div>
      <button class="bgsw" data-bg="checker" aria-pressed="true" aria-label="Checkerboard background" style="background:repeating-conic-gradient(#d8dae0 0 25%,#fff 0 50%) 0 0/10px 10px"></button>
      <button class="bgsw" data-bg="white" aria-pressed="false" aria-label="White background" style="background:#fff"></button>
      <button class="bgsw" data-bg="dark" aria-pressed="false" aria-label="Dark background" style="background:#1b1d22"></button>
    </div>
    <div class="float" id="zoombar">
      <button class="ico" id="zOut" aria-label="Zoom out">−</button>
      <span class="zlabel" id="zLabel">100%</span>
      <button class="ico" id="zIn" aria-label="Zoom in">+</button>
      <button class="btn" id="zFit" style="padding:4px 10px">Fit</button>
    </div>
    <div class="float" id="statusbar"><span class="spin"></span><span id="status" role="status" aria-live="polite"></span></div>
  </div>

  <aside>
    <div class="sec">
      <h2>Colors <button class="link" id="redetect">Detect again</button></h2>
      <div id="palette"></div>
      <button class="btn pickbtn" id="pickBtn" aria-pressed="false">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m2 22 1-1h3l9-9"/><path d="M3 21v-3l9-9"/><path d="m15 6 3.4-3.4a2.1 2.1 0 1 1 3 3L18 9l.4.4a2.1 2.1 0 1 1-3 3l-3.8-3.8a2.1 2.1 0 1 1 3-3l.4.4Z"/></svg>
        Pick a color from the image
      </button>
      <p class="note">Everything set to Drop is removed from the trace. Pixels are matched to the nearest color, so soft edges land on the right side.</p>
    </div>
    <div class="sec"><h2>Edges</h2><div id="gEdges"></div>
      <p class="note">Fill weight grows or shrinks filled regions before tracing. Path simplification removes tiny turns. Curve averaging smooths points along each contour as a final polish pass.</p></div>
    <div class="sec"><h2>Cleanup</h2><div id="gClean"></div>
      <p class="note">Dark grey cutoff applies when a black or grey color is set to Drop.</p></div>
    <div class="sec"><h2>Print <button class="link" id="reset">Reset all</button></h2><div id="gPrint"></div>
      <p class="note">Trapping grows the lower layers underneath so print misregistration doesn’t leave white gaps.</p></div>
    <div class="sec">
      <h2>Quality</h2>
      <div class="quality">
        <div class="seg" id="qSeg">
          <button data-q="1" aria-pressed="false">Draft 1×</button>
          <button data-q="2" aria-pressed="true">Standard 2×</button>
          <button data-q="4" aria-pressed="false">High 4×</button>
        </div>
      </div>
      <p class="note">Higher quality traces at a larger size for smoother curves. It takes longer and makes bigger files.</p>
    </div>
    <div class="sec" style="border-bottom:0">
      <h2>Download</h2>
      <div class="dl">
        <button class="btn primary big" id="dlSvg">Download SVG</button>
        <button class="btn" id="dlPng">PNG</button>
        <button class="btn" id="dlPdf">PDF</button>
        <button class="btn" id="dlCopy">Copy SVG</button>
      </div>
    </div>
  </aside>
</main>

<script>
"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const DEFAULTS = { corner: 28, thick: 0.5, sigma: 0.125, speckle: 0.25, simplify: 0.35, polish: 0.0, dark: 110, alpha: 128, trap: 0 };
const SLIDERS = [
  { g: "gEdges", k: "corner",  label: "Corners",            min: 5, max: 90, step: 1,    lo: "Sharp",    hi: "Smooth",    fmt: v => v + "°" },
  { g: "gEdges", k: "thick",   label: "Fill weight",        min: 0, max: 1,  step: .01,  lo: "Lighter",  hi: "Heavier",   fmt: v => (v >= .5 ? "+" : "") + Math.round((v - .5) * 100) + "%" },
  { g: "gEdges", k: "sigma",   label: "Edge softness",      min: 0, max: .5, step: .005, lo: "Crisp",    hi: "Soft",      fmt: v => v.toFixed(2) + " px" },
  { g: "gEdges", k: "speckle", label: "Speckle cleanup",    min: 0, max: 8,  step: .25,  lo: "Keep all", hi: "More cleanup",  fmt: v => v.toFixed(2) + " px" },
  { g: "gEdges", k: "simplify",label: "Path simplification",min: 0, max: 1,  step: .01,  lo: "More detail", hi: "Smoother", fmt: v => Math.round(v * 100) + "%" },
  { g: "gEdges", k: "polish",  label: "Curve averaging",    min: 0, max: 1,  step: .01,  lo: "Off",      hi: "Maximum",   fmt: v => Math.round(v * 100) + "%" },
  { g: "gClean", k: "dark",    label: "Dark grey cutoff",   min: 0, max: 200, step: 1,   lo: "Black only", hi: "Dark greys", fmt: v => String(v) },
  { g: "gClean", k: "alpha",   label: "Edge transparency",  min: 1, max: 254, step: 1,   lo: "Keep more", hi: "Keep less", fmt: v => String(v) },
  { g: "gPrint", k: "trap",    label: "Trapping",           min: 0, max: 3,  step: .25,  lo: "None",     hi: "Heavy",     fmt: v => v.toFixed(2) + " px" },
];

const S = {
  id: null, name: "", w: 0, h: 0, pdf: false,
  palette: [], params: { ...DEFAULTS }, scale: 2,
  svg: "", svgUrl: "",
  view: "split", split: 0.5, z: 1, px: 0, py: 0, fit: 1,
  pick: false, ctrl: null, timer: null, ctx: null,
};

/* ---------- loading ---------- */
async function load(file) {
  if (!file) return;
  setStatus("Reading image…", true);
  show(true);
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Upload failed.");
    Object.assign(S, { id: j.id, w: j.w, h: j.h, pdf: j.pdf, palette: j.palette, name: file.name || "pasted-image" });
    S.params = { ...DEFAULTS };
    $("#fname").textContent = S.name;
    $("#dlPdf").disabled = !S.pdf;
    $("#dlPdf").title = S.pdf ? "" : "Install cairosvg to enable PDF export";
    renderSliders();
    renderPalette();
    await setOriginal();
    trace();
  } catch (e) {
    show(false);
    alert(e.message);
  }
}

function setOriginal() {
  return new Promise((res) => {
    const img = $("#imgOrig");
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = S.w; c.height = S.h;
      S.ctx = c.getContext("2d", { willReadFrequently: true });
      S.ctx.drawImage(img, 0, 0);
      for (const el of [img, $("#imgVec")]) { el.style.width = S.w + "px"; el.style.height = S.h + "px"; }
      fitView();
      res();
    };
    img.src = "/api/image/" + S.id;
  });
}

function show(on) {
  $("#empty").style.display = on ? "none" : "grid";
  $("#app").classList.toggle("on", on);
  $("#newBtn").hidden = !on;
  if (!on) { $("#fname").textContent = ""; }
}

/* ---------- tracing ---------- */
function params() {
  const p = S.params;
  return { scale: S.scale, corner: p.corner, thresh: 0.70 - 0.28 * p.thick, sigma: p.sigma,
           speckle: p.speckle, simplify: p.simplify, polish: p.polish,
           dark: p.dark, alpha: p.alpha, trap: p.trap };
}

function schedule(ms = 320) { clearTimeout(S.timer); S.timer = setTimeout(trace, ms); }

async function trace() {
  if (!S.id) return;
  if (S.ctrl) S.ctrl.abort();
  const ctrl = (S.ctrl = new AbortController());
  setStatus("Tracing…", true);
  try {
    const r = await fetch("/api/trace", {
      method: "POST", signal: ctrl.signal, headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: S.id, palette: S.palette, params: params() }),
    });
    if (!r.ok) { const j = await r.json(); throw new Error(j.error || "Tracing failed."); }
    const svg = await r.text();
    if (ctrl.signal.aborted) return;
    S.svg = svg;
    if (S.svgUrl) URL.revokeObjectURL(S.svgUrl);
    S.svgUrl = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    $("#imgVec").src = S.svgUrl;
    const kb = Math.round(svg.length / 1024);
    setStatus(`${Number(r.headers.get("X-Paths")).toLocaleString()} paths · ${kb.toLocaleString()} KB · ${r.headers.get("X-Seconds")} s`, false);
  } catch (e) {
    if (e.name === "AbortError") return;
    setStatus(e.message, false, true);
  }
}

function setStatus(msg, busy, isErr) {
  $("#statusbar").classList.toggle("busy", !!busy);
  const el = $("#status");
  el.textContent = msg;
  el.classList.toggle("err", !!isErr);
}

/* ---------- sidebar ---------- */
function renderSliders() {
  for (const g of ["gEdges", "gClean", "gPrint"]) $("#" + g).innerHTML = "";
  for (const s of SLIDERS) {
    const d = document.createElement("div");
    d.className = "ctl";
    d.innerHTML = `<div class="row"><label for="s_${s.k}">${s.label}</label><span class="val" id="v_${s.k}"></span></div>
      <input type="range" id="s_${s.k}" min="${s.min}" max="${s.max}" step="${s.step}">
      <div class="ends"><span>${s.lo}</span><span>${s.hi}</span></div>`;
    $("#" + s.g).appendChild(d);
    const inp = d.querySelector("input");
    inp.value = S.params[s.k];
    $("#v_" + s.k).textContent = s.fmt(+inp.value);
    inp.addEventListener("input", () => {
      S.params[s.k] = +inp.value;
      $("#v_" + s.k).textContent = s.fmt(+inp.value);
      schedule(380);
    });
    inp.addEventListener("change", () => schedule(0));
  }
}

function renderPalette() {
  const root = $("#palette");
  root.innerHTML = "";
  S.palette.forEach((c, i) => {
    const row = document.createElement("div");
    row.className = "sw" + (c.keep ? "" : " dropped");
    row.innerHTML = `
      <span class="chip" style="background:${c.hex}"></span>
      <span><span class="hex">${c.hex.toUpperCase()}</span><span class="pct">${c.pct == null ? "picked" : c.pct + "% of image"}</span></span>
      <span class="seg" role="group" aria-label="${c.hex}">
        <button class="on-keep" data-i="${i}" data-keep="1" aria-pressed="${c.keep}">Keep</button>
        <button class="on-drop" data-i="${i}" data-keep="0" aria-pressed="${!c.keep}">Drop</button>
      </span>
      <button class="x" data-rm="${i}" aria-label="Remove ${c.hex}" title="Remove">×</button>`;
    root.appendChild(row);
  });
}

$("#palette").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.rm != null) {
    if (S.palette.length <= 2) return;
    S.palette.splice(+b.dataset.rm, 1);
  } else if (b.dataset.i != null) {
    S.palette[+b.dataset.i].keep = b.dataset.keep === "1";
  } else return;
  renderPalette();
  schedule(120);
});

$("#redetect").addEventListener("click", async () => {
  if (!S.id) return;
  const j = await (await fetch("/api/palette/" + S.id)).json();
  if (j.palette) { S.palette = j.palette; renderPalette(); schedule(0); }
  else setStatus(j.error || "Couldn’t detect colors", false, true);
});

$("#pickBtn").addEventListener("click", () => setPick(!S.pick));
function setPick(on) {
  S.pick = on;
  $("#pickBtn").setAttribute("aria-pressed", on);
  $("#stage").classList.toggle("pick", on);
  if (on) setStatus("Click a color in the image", false);
}

$("#reset").addEventListener("click", () => { S.params = { ...DEFAULTS }; renderSliders(); schedule(0); });

$$("#qSeg button").forEach((b) => b.addEventListener("click", () => {
  S.scale = +b.dataset.q;
  $$("#qSeg button").forEach((x) => x.setAttribute("aria-pressed", x === b));
  schedule(0);
}));

/* ---------- viewer ---------- */
const stage = $("#stage");
function applyView() {
  const t = `translate(${S.px}px,${S.py}px) scale(${S.z})`;
  $$(".world").forEach((w) => (w.style.transform = t));
  $("#zLabel").textContent = Math.round(S.z * 100) + "%";
  const pct = S.split * 100;
  const split = S.view === "split";
  $("#layerOrig").style.clipPath = split ? `inset(0 ${100 - pct}% 0 0)` : "none";
  $("#layerVec").style.clipPath = split ? `inset(0 0 0 ${pct}%)` : "none";
  $("#layerOrig").style.display = S.view === "vec" ? "none" : "";
  $("#layerVec").style.display = S.view === "orig" ? "none" : "";
  $("#divider").style.left = pct + "%";
  $("#divider").setAttribute("aria-valuenow", Math.round(pct));
  stage.dataset.view = S.view;
}

function fitView() {
  const W = stage.clientWidth, H = stage.clientHeight;
  S.fit = Math.min(W / S.w, H / S.h) * 0.9;
  S.z = S.fit;
  S.px = (W - S.w * S.z) / 2;
  S.py = (H - S.h * S.z) / 2;
  applyView();
}

function zoomAt(cx, cy, f) {
  const r = stage.getBoundingClientRect();
  const x = cx - r.left, y = cy - r.top;
  const nz = Math.min(40, Math.max(S.fit * 0.2, S.z * f));
  const k = nz / S.z;
  S.px = x - (x - S.px) * k;
  S.py = y - (y - S.py) * k;
  S.z = nz;
  applyView();
}

stage.addEventListener("wheel", (e) => { e.preventDefault(); zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
$("#zIn").onclick = () => { const r = stage.getBoundingClientRect(); zoomAt(r.left + r.width / 2, r.top + r.height / 2, 1.4); };
$("#zOut").onclick = () => { const r = stage.getBoundingClientRect(); zoomAt(r.left + r.width / 2, r.top + r.height / 2, 1 / 1.4); };
$("#zFit").onclick = fitView;
window.addEventListener("resize", () => S.id && fitView());

let drag = null;
stage.addEventListener("pointerdown", (e) => {
  if (e.target.closest(".float") || e.target.closest("#divider")) return;
  drag = { x: e.clientX, y: e.clientY, px: S.px, py: S.py, moved: false };
  stage.setPointerCapture(e.pointerId);
  stage.classList.add("dragging");
});
stage.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
  S.px = drag.px + dx; S.py = drag.py + dy;
  applyView();
});
stage.addEventListener("pointerup", (e) => {
  stage.classList.remove("dragging");
  if (drag && !drag.moved && S.pick) pickAt(e.clientX, e.clientY);
  drag = null;
});

function pickAt(cx, cy) {
  const r = stage.getBoundingClientRect();
  const x = Math.floor((cx - r.left - S.px) / S.z), y = Math.floor((cy - r.top - S.py) / S.z);
  if (x < 0 || y < 0 || x >= S.w || y >= S.h) return;
  const [R, G, B, A] = S.ctx.getImageData(x, y, 1, 1).data;
  if (A < 128) { setStatus("That spot is transparent. Try another.", false); return; }
  const hex = "#" + [R, G, B].map((v) => v.toString(16).padStart(2, "0")).join("");
  if (!S.palette.some((c) => c.hex === hex)) S.palette.push({ hex, pct: null, keep: true });
  renderPalette();
  setPick(false);
  schedule(0);
}

let dv = false;
$("#divider").addEventListener("pointerdown", (e) => { dv = true; $("#divider").setPointerCapture(e.pointerId); });
$("#divider").addEventListener("pointermove", (e) => {
  if (!dv) return;
  const r = stage.getBoundingClientRect();
  S.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  applyView();
});
$("#divider").addEventListener("pointerup", () => (dv = false));
$("#divider").addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") S.split = Math.max(0, S.split - 0.02);
  else if (e.key === "ArrowRight") S.split = Math.min(1, S.split + 0.02);
  else return;
  e.preventDefault(); applyView();
});

$$("#viewSeg button").forEach((b) => b.addEventListener("click", () => {
  S.view = b.dataset.view;
  $$("#viewSeg button").forEach((x) => x.setAttribute("aria-pressed", x === b));
  applyView();
}));
$$(".bgsw").forEach((b) => b.addEventListener("click", () => {
  stage.dataset.bg = b.dataset.bg;
  $$(".bgsw").forEach((x) => x.setAttribute("aria-pressed", x === b));
}));

window.addEventListener("keydown", (e) => {
  if (!S.id || e.target.matches("input,textarea")) return;
  if (e.key === "+" || e.key === "=") $("#zIn").click();
  else if (e.key === "-") $("#zOut").click();
  else if (e.key === "0") fitView();
  else if (e.key === "Escape") setPick(false);
});

/* ---------- downloads ---------- */
function save(blob, ext) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (S.name.replace(/\.[^.]+$/, "") || "trace") + "." + ext;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}
$("#dlSvg").onclick = () => S.svg && save(new Blob([S.svg], { type: "image/svg+xml" }), "svg");
$("#dlCopy").onclick = async () => {
  if (!S.svg) return;
  await navigator.clipboard.writeText(S.svg);
  setStatus("SVG copied to clipboard", false);
};
$("#dlPng").onclick = () => {
  if (!S.svgUrl) return;
  const img = new Image();
  img.onload = () => {
    const k = Math.min(2, 8000 / Math.max(S.w, S.h));
    const c = document.createElement("canvas");
    c.width = Math.round(S.w * k); c.height = Math.round(S.h * k);
    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
    c.toBlob((b) => save(b, "png"), "image/png");
  };
  img.src = S.svgUrl;
};
$("#dlPdf").onclick = async () => {
  if (!S.svg || !S.pdf) return;
  setStatus("Building PDF…", true);
  const r = await fetch("/api/pdf", { method: "POST", body: S.svg });
  if (!r.ok) { setStatus("PDF export failed", false, true); return; }
  save(await r.blob(), "pdf");
  setStatus("PDF saved", false);
};

/* ---------- input: choose, drop, paste ---------- */
const fileInput = $("#file");
$("#chooseBtn").onclick = (e) => { e.stopPropagation(); fileInput.click(); };
$("#drop").onclick = () => fileInput.click();
$("#drop").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
fileInput.onchange = () => { load(fileInput.files[0]); fileInput.value = ""; };
$("#newBtn").onclick = () => { S.id = null; show(false); };
window.addEventListener("dragover", (e) => { e.preventDefault(); $("#drop").classList.add("over"); });
window.addEventListener("dragleave", () => $("#drop").classList.remove("over"));
window.addEventListener("drop", (e) => { e.preventDefault(); $("#drop").classList.remove("over"); load(e.dataTransfer.files[0]); });
window.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])][0]; if (f) load(f); });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    port = 5757
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"Dropout running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=port, threaded=True)
