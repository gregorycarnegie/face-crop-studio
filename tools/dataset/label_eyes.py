"""Click two eye points per face, to turn Open Images boxes into training labels.

  label_eyes.py <data_dir> [--port 8765] [--count 2000]

Open Images has hand-drawn face boxes but no landmarks, and `fcs-core::face_cropper` aligns
crops on the eyes, so a replacement detector needs eye points. This serves the boxes one at a
time, zoomed, and records two clicks each.

Labels append to <data_dir>/eye_labels.jsonl as you go, so a refresh, a crash or a week off
loses nothing: on restart, anything already answered is dropped from the queue. The last line
for a face wins, which is what makes "go back" work.

Coordinates are stored normalised (0-1 of image width/height) so they stay valid whatever
size the image is served at. Clicks are recorded as viewer-left and viewer-right, matching
what the labeller sees; YuNet's own order is the mirror of that (landmarks[0] is the eye on
the viewer's right), and `export` below emits its order.

ponytail: single user, no auth, binds to localhost only. If two people ever label at once,
give each their own port and jsonl and merge afterwards.
"""

import argparse
import csv
import json
import random
import re
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

IMAGE_ID = re.compile(r"^[0-9a-f]{16}$")
MIN_WIDTH_AT_640 = 32 / 640  # faces too small to crop are not worth labelling


def build_queue(data: Path, count: int) -> list[dict]:
    faces = []
    for split in ("validation", "test"):
        path = data / f"faces-{split}-annotations-bbox.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["IsGroupOf"] == "1" or row["IsDepiction"] == "1":
                    continue
                xmin, xmax = float(row["XMin"]), float(row["XMax"])
                if xmax - xmin < MIN_WIDTH_AT_640:
                    continue
                if not (data / "images" / f"{row['ImageID']}.jpg").exists():
                    continue
                faces.append({
                    "id": f"{row['ImageID']}:{xmin:.6f},{float(row['YMin']):.6f}",
                    "image": row["ImageID"],
                    "box": [xmin, float(row["YMin"]), xmax, float(row["YMax"])],
                })
    random.Random(0).shuffle(faces)  # seeded: the same queue every run
    return faces[:count]


def load_done(path: Path) -> OrderedDict:
    done = OrderedDict()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["id"]] = rec  # last line for an id wins
    return done


PAGE = """<!doctype html><meta charset=utf-8><title>Eye labelling</title>
<style>
 body{font:14px system-ui;margin:0;background:#111;color:#eee;display:flex;
      flex-direction:column;align-items:center;gap:12px;padding:16px}
 canvas{border-radius:6px;cursor:crosshair;background:#000}
 #bar{display:flex;gap:16px;align-items:center}
 kbd{background:#333;border-radius:3px;padding:1px 5px;font-family:inherit}
 #hint{color:#9ad;min-height:1.2em} #done{color:#888}
</style>
<div id=bar><strong id=hint>loading…</strong><span id=done></span></div>
<canvas id=c width=560 height=560></canvas>
<div><kbd>click</kbd> left eye then right eye · <kbd>u</kbd> undo click ·
 <kbd>s</kbd> skip (not a usable face) · <kbd>b</kbd> back</div>
<script>
const c = document.getElementById('c'), ctx = c.getContext('2d');
const hint = document.getElementById('hint'), doneEl = document.getElementById('done');
let queue = [], at = 0, pts = [], img = new Image(), view = null, total = 0, busy = false;

fetch('queue.json').then(r => r.json()).then(q => {
  queue = q.queue; total = q.total; show();
});

function show() {
  pts = [];
  if (at >= queue.length) {
    hint.textContent = queue.length ? 'queue finished — thank you' : 'nothing left to label';
    ctx.clearRect(0, 0, c.width, c.height);
    view = null;
    return;
  }
  const item = queue[at];
  doneEl.textContent = `${total - queue.length + at} of ${total} answered`;
  const ready = () => { measure(item); render(); };
  img.onload = ready;
  const src = 'img/' + item.image + '.jpg';
  if (img.getAttribute('src') === src && img.complete && img.naturalWidth) {
    ready();                       // consecutive faces in one photo: onload will not fire
  } else {
    img.setAttribute('src', src);
  }
}

function measure(item) {
  const [x0, y0, x1, y1] = item.box;
  const iw = img.naturalWidth, ih = img.naturalHeight;
  const bw = (x1 - x0) * iw, bh = (y1 - y0) * ih;
  const side = Math.max(bw, bh) * 1.9;            // box plus context around it
  view = {sx: x0 * iw + bw / 2 - side / 2, sy: y0 * ih + bh / 2 - side / 2,
          side, iw, ih, bw, bh, bx: x0 * iw, by: y0 * ih};
}

function render() {
  if (!view) return;
  const k = c.width / view.side;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(img, view.sx, view.sy, view.side, view.side, 0, 0, c.width, c.height);
  ctx.strokeStyle = '#4af'; ctx.lineWidth = 2;
  ctx.strokeRect((view.bx - view.sx) * k, (view.by - view.sy) * k, view.bw * k, view.bh * k);
  for (const [i, p] of pts.entries()) {
    ctx.fillStyle = i ? '#fa4' : '#4f8';
    ctx.beginPath();
    ctx.arc((p[0] * view.iw - view.sx) * k, (p[1] * view.ih - view.sy) * k, 4, 0, 7);
    ctx.fill();
  }
  hint.textContent = pts.length === 0 ? 'click the LEFT eye (as you see it)'
                   : pts.length === 1 ? 'now the RIGHT eye' : 'saving…';
}

c.onclick = e => {
  if (!view || pts.length >= 2 || busy) return;
  const r = c.getBoundingClientRect(), k = view.side / c.width;
  pts.push([(view.sx + (e.clientX - r.left) * k) / view.iw,
            (view.sy + (e.clientY - r.top) * k) / view.ih]);
  render();
  if (pts.length === 2) save({viewer_left: pts[0], viewer_right: pts[1]});
};

function save(extra) {
  if (busy || at >= queue.length) return;
  busy = true;
  const item = queue[at];
  fetch('save', {method: 'POST', body: JSON.stringify(
    Object.assign({id: item.id, image: item.image, box: item.box}, extra))})
    .then(() => { busy = false; at++; show(); })
    .catch(() => { busy = false; hint.textContent = 'save failed — is the server still up?'; });
}

addEventListener('keydown', e => {
  if (e.key === 'u') { pts.pop(); render(); }
  else if (e.key === 's') save({skipped: true});
  else if (e.key === 'b' && at > 0) { at--; show(); }   // re-answering overwrites: last line wins
});
</script>
"""


class Handler(BaseHTTPRequestHandler):
    data: Path
    queue: list[dict]
    out: Path
    total: int

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/queue.json":
            body = json.dumps({"queue": self.queue, "total": self.total}).encode("utf-8")
            return self._send(200, body, "application/json")
        if path.startswith("/img/"):
            stem = path[len("/img/"):].removesuffix(".jpg")
            if not IMAGE_ID.match(stem):
                return self._send(404, b"no", "text/plain")
            file = self.data / "images" / f"{stem}.jpg"
            if not file.exists():
                return self._send(404, b"no", "text/plain")
            return self._send(200, file.read_bytes(), "image/jpeg")
        self._send(404, b"no", "text/plain")

    def do_POST(self):
        if self.path != "/save":
            return self._send(404, b"no", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        rec = json.loads(self.rfile.read(length) or b"{}")
        with open(self.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self._send(200, b"ok", "text/plain")

    def log_message(self, format, *args):  # noqa: A002 - matches BaseHTTPRequestHandler
        pass  # one line per served image is just noise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--count", type=int, default=2000)
    args = ap.parse_args()

    data = Path(args.data_dir).resolve()
    out = data / "eye_labels.jsonl"
    queue = build_queue(data, args.count)
    done = load_done(out)
    remaining = [f for f in queue if f["id"] not in done]

    Handler.data, Handler.queue, Handler.out, Handler.total = data, remaining, out, len(queue)
    labelled = sum(1 for r in done.values() if not r.get("skipped"))
    # flush: stdout is block-buffered when redirected, and the URL below is the whole point.
    print(f"{len(queue)} faces in the queue, {len(done)} already answered "
          f"({labelled} with eye points), {len(remaining)} to go", flush=True)
    print(f"labels append to {out}", flush=True)
    print(f"open http://127.0.0.1:{args.port}/  (ctrl-c to stop; progress is saved)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
