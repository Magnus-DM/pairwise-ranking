#!/usr/bin/env python3
"""
ranker.py - a tiny private web app for ranking the entries of a dataset by
answering "which do you prefer, A or B?" on your phone.

Only uses the Python standard library (Python 3.8+). Nothing to install.

USAGE
    python3 ranker.py data.csv
    python3 ranker.py data.json --title name --show "year,price,notes" --image photo

OPTIONS
    --id FIELD      column that uniquely identifies an entry
                    (default: a column called "id", otherwise the row number)
    --title FIELD   column used as the heading of each card (default: first column)
    --show F1,F2    extra columns shown on each card (default: the next 5 columns)
    --image FIELD   column with an image URL, or a file path relative to the data file
                    (default: a column called image/photo/picture/thumbnail if present)
    --db PATH       where your choices are stored (default: <data file>.prefs.sqlite)
    --name TEXT     app name shown on the home screen (default: data file name)
    --host / --port default 127.0.0.1:8000

DATA FORMATS
    .csv / .tsv     header row + one entry per line (delimiter auto-detected)
    .json           a list of objects, or an object mapping id -> object
    .jsonl          one JSON object per line

ON YOUR PHONE
    The server listens on 127.0.0.1 by default. To use it from your phone,
    expose it privately (for example with a VPN such as Tailscale, or
    --host 0.0.0.0 on a trusted home network), open the address in the
    browser and add it to the home screen.

HOW THE RANKING WORKS
    Every choice is stored as a row in a SQLite file. Scores are fitted with a
    Bradley-Terry model (the statistics behind Elo, but order-independent),
    shown on an Elo-like scale where 1500 = average. The "±" is a rough 95%
    uncertainty. Pairs are picked to be informative: the first entry is the
    one whose score is least certain, and its opponent is the one whose
    outcome is hardest to predict (a close, rarely-seen matchup). A solid
    ranking typically needs roughly N * log2(N) comparisons (about 1,100 for
    150 entries), but the top of the list settles much sooner.
"""

import argparse
import csv
import html
import io
import json
import math
import mimetypes
import random
import sqlite3
import struct
import sys
import threading
import time
import zlib
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

SCALE = 400 / math.log(10)  # natural-log strength -> Elo-style points
IMAGE_NAMES = ("image", "img", "photo", "picture", "thumbnail", "image_url", "cover")


# --------------------------------------------------------------------------- data


def read_rows(path):
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")
    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list)]
            if len(lists) == 1:
                data = lists[0]
            elif data and all(isinstance(v, dict) for v in data.values()):
                data = [{"id": k, **v} for k, v in data.items()]
            else:
                sys.exit("JSON must be a list of objects, or an object mapping ids to objects.")
        if not isinstance(data, list):
            sys.exit("JSON must be a list of objects.")
        return [r if isinstance(r, dict) else {"value": r} for r in data]
    if suffix in (".jsonl", ".ndjson"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    first_line = text.split("\n", 1)[0]
    delimiter = max(",;\t|", key=first_line.count)
    return [
        {k: v for k, v in r.items() if k is not None}
        for r in csv.DictReader(io.StringIO(text), delimiter=delimiter)
    ]


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


class Dataset:
    def __init__(self, path, id_field=None, title_field=None, show=None, image_field=None):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            sys.exit(f"Data file not found: {self.path}")
        rows = read_rows(self.path)
        if not rows:
            sys.exit(f"No entries found in {self.path.name}")

        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        lower = {str(c).lower(): c for c in cols}

        def check(field, flag):
            if field and field not in cols:
                sys.exit(
                    f'--{flag} "{field}" is not a column. Columns: {", ".join(map(str, cols))}'
                )
            return field

        id_field = check(id_field, "id") or lower.get("id")
        image_field = check(image_field, "image") or next(
            (lower[n] for n in IMAGE_NAMES if n in lower), None
        )
        rest = [c for c in cols if c not in (id_field, image_field)]
        title_field = check(title_field, "title") or (rest[0] if rest else id_field)
        if show is None:
            show_fields = [c for c in rest if c != title_field][:5]
        else:
            show_fields = [check(c.strip(), "show") for c in show.split(",") if c.strip()]

        self.columns, self.id_field, self.title_field = cols, id_field, title_field
        self.show_fields, self.image_field = show_fields, image_field
        self.items, self.order, self.local_images = {}, [], {}

        for n, r in enumerate(rows, 1):
            row = {k: to_text(r.get(k)) for k in cols}
            iid = row[id_field] if id_field else str(n)
            if not iid:
                sys.exit(f'Entry {n} has an empty "{id_field}" value.')
            if iid in self.items:
                sys.exit(f'Duplicate id "{iid}". Pick a column with unique values using --id.')
            image = row.get(image_field, "") if image_field else ""
            if image and not image.startswith(("http://", "https://", "data:")):
                self.local_images[iid] = (self.path.parent / image).resolve()
                image = "/img/" + quote(iid, safe="")
            self.items[iid] = {
                "row": row,
                "title": row.get(title_field) or iid,
                "image": image or None,
                "fields": [[f, row[f]] for f in show_fields if row.get(f)],
            }
            self.order.append(iid)


# --------------------------------------------------------------------------- ranking


class Ranker:
    def __init__(self, ds, db_path):
        self.ds = ds
        self.ids = ds.order
        self.idx = {iid: k for k, iid in enumerate(self.ids)}
        self.lock = threading.RLock()
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            a TEXT NOT NULL, b TEXT NOT NULL,
            result REAL NOT NULL,          -- 1 = a preferred, 0 = b preferred, 0.5 = equal
            created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        self.db.commit()
        self.rows = list(self.db.execute("SELECT id, a, b, result FROM comparisons ORDER BY id"))
        self.recent = deque(maxlen=8)
        self.logp = [0.0] * len(self.ids)
        self.dirty = True
        self._refit(time_budget=30)

    def _refit(self, time_budget=1.0):
        """Fit Bradley-Terry strengths with the MM algorithm, warm-started.

        Each entry gets one virtual win and one virtual loss against a fixed
        average opponent, which keeps unbeaten / winless entries finite.
        """
        n = len(self.ids)
        wins = [1.0] * n
        record = [[0, 0, 0] for _ in range(n)]
        pair_counts = defaultdict(int)
        total = 0
        for _, a, b, r in self.rows:
            i, j = self.idx.get(a), self.idx.get(b)
            if i is None or j is None or i == j:
                continue
            total += 1
            wins[i] += r
            wins[j] += 1 - r
            pair_counts[(i, j) if i < j else (j, i)] += 1
            if r > 0.75:
                record[i][0] += 1
                record[j][1] += 1
            elif r < 0.25:
                record[i][1] += 1
                record[j][0] += 1
            else:
                record[i][2] += 1
                record[j][2] += 1

        neighbours = [[] for _ in range(n)]
        for (i, j), c in pair_counts.items():
            neighbours[i].append((j, c))
            neighbours[j].append((i, c))

        p = [math.exp(x) for x in self.logp]
        deadline = time.monotonic() + time_budget
        for _ in range(2000):
            delta = 0.0
            for i in range(n):
                denom = 2.0 / (p[i] + 1.0)
                pi = p[i]
                for j, c in neighbours[i]:
                    denom += c / (pi + p[j])
                new = wins[i] / denom
                delta = max(delta, abs(math.log(new / pi)))
                p[i] = new
            if delta < 1e-5 or time.monotonic() > deadline:
                break

        se = []
        for i in range(n):
            info = 2.0 * p[i] / (p[i] + 1.0) ** 2
            for j, c in neighbours[i]:
                info += c * p[i] * p[j] / (p[i] + p[j]) ** 2
            se.append(1.0 / math.sqrt(info))

        self.logp = [math.log(x) for x in p]
        self.se, self.record, self.pair_counts, self.total = se, record, pair_counts, total
        self.seen = sum(1 for r in record if any(r))
        self.dirty = False

    def _fresh(self):
        if self.dirty:
            self._refit()

    def stats(self):
        with self.lock:
            self._fresh()
            return {"comparisons": self.total, "seen": self.seen, "total": len(self.ids)}

    def card(self, iid):
        it = self.ds.items[iid]
        return {"id": iid, "title": it["title"], "image": it["image"], "fields": it["fields"]}

    def next_pair(self):
        with self.lock:
            self._fresh()
            n = len(self.ids)
            if n < 2:
                return None
            # First entry: the one we know least about (fewest / least decisive comparisons).
            pool = random.sample(range(n), min(n, 40))
            i = max(pool, key=lambda k: self.se[k] + 0.05 * random.random())
            # Second entry: a close, rarely-seen matchup, avoiding pairs shown just now.
            recent = set(self.recent)
            best, best_gain = None, -math.inf
            for k in random.sample(range(n), min(n, 120)):
                if k == i:
                    continue
                q = 1.0 / (1.0 + math.exp(self.logp[k] - self.logp[i]))
                repeats = self.pair_counts.get((min(i, k), max(i, k)), 0)
                gain = q * (1 - q) * (1 + self.se[k]) / (1 + repeats) ** 2
                if frozenset((i, k)) in recent:
                    gain -= 10
                gain += 0.02 * random.random()
                if gain > best_gain:
                    best, best_gain = k, gain
            self.recent.append(frozenset((i, best)))
            a, b = (i, best) if random.random() < 0.5 else (best, i)
            return [self.card(self.ids[a]), self.card(self.ids[b])]

    def vote(self, a, b, result):
        with self.lock:
            if a not in self.idx or b not in self.idx or a == b:
                raise ValueError("unknown entries")
            if result not in (0, 0.5, 1):
                raise ValueError("result must be 1, 0 or 0.5")
            cur = self.db.execute(
                "INSERT INTO comparisons (a, b, result) VALUES (?, ?, ?)", (a, b, float(result))
            )
            self.db.commit()
            self.rows.append((cur.lastrowid, a, b, float(result)))
            self.dirty = True

    def undo(self):
        with self.lock:
            if not self.rows:
                return False, None
            rid, a, b, _ = self.rows.pop()
            self.db.execute("DELETE FROM comparisons WHERE id = ?", (rid,))
            self.db.commit()
            self.dirty = True
            if a in self.idx and b in self.idx:
                return True, [self.card(a), self.card(b)]
            return True, None

    def ranking(self):
        with self.lock:
            if self.dirty:
                self._refit(time_budget=5)
            order = sorted(range(len(self.ids)), key=lambda k: -self.logp[k])
            out = []
            for rank, k in enumerate(order, 1):
                iid = self.ids[k]
                wins, losses, ties = self.record[k]
                out.append(
                    {
                        "rank": rank,
                        "id": iid,
                        "title": self.ds.items[iid]["title"],
                        "image": self.ds.items[iid]["image"],
                        "rating": round(1500 + SCALE * self.logp[k]),
                        "pm": round(1.96 * SCALE * self.se[k]),
                        "w": wins,
                        "l": losses,
                        "t": ties,
                        "n": wins + losses + ties,
                    }
                )
            return out

    def ranking_csv(self):
        buf = io.StringIO()
        writer = csv.writer(buf)
        extra = list(self.ds.columns)
        writer.writerow(["rank", "rating", "plus_minus", "wins", "losses", "equal"] + extra)
        for r in self.ranking():
            row = self.ds.items[r["id"]]["row"]
            writer.writerow(
                [r["rank"], r["rating"], r["pm"], r["w"], r["l"], r["t"]]
                + [row.get(c, "") for c in extra]
            )
        return buf.getvalue()


# --------------------------------------------------------------------------- web app

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="__SHORT__">
<meta name="theme-color" content="#f4f5f7" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#111318" media="(prefers-color-scheme: dark)">
<link rel="apple-touch-icon" href="/icon.png">
<link rel="manifest" href="/manifest.webmanifest">
<title>__TITLE__</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--ink:#16181d;--muted:#6b7280;--line:#e3e5ea;--accent:#4f6bff}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--card:#1b1e25;--ink:#eceef2;--muted:#9aa1ad;--line:#2a2e37;--accent:#7b90ff}}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
[hidden]{display:none!important}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
  font:16px/1.35 -apple-system,BlinkMacSystemFont,"SF Pro Text",system-ui,sans-serif;touch-action:manipulation}
body{display:flex;flex-direction:column;height:100dvh;
  padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)}
header{display:flex;align-items:center;gap:8px;padding:10px 16px 6px}
.tabs{display:flex;background:var(--line);border-radius:9px;padding:2px}
.tabs button{border:0;background:transparent;color:var(--ink);font:inherit;font-size:14px;font-weight:600;padding:6px 14px;border-radius:7px}
.tabs button.on{background:var(--card);box-shadow:0 1px 2px rgba(0,0,0,.15)}
.stat{margin-left:auto;font-size:13px;color:var(--muted);font-variant-numeric:tabular-nums;text-align:right}
main{flex:1;min-height:0;display:flex;flex-direction:column}
#compare{flex:1;min-height:0;display:flex;flex-direction:column;gap:10px;padding:6px 16px 10px}
.card{flex:1 1 0;min-height:0;display:flex;flex-direction:column;background:var(--card);border:1px solid var(--line);
  border-radius:16px;overflow:hidden;cursor:pointer;user-select:none;-webkit-user-select:none;
  transition:transform .12s ease,box-shadow .12s ease,opacity .15s ease}
.card:active{transform:scale(.985)}
.card.picked{box-shadow:0 0 0 3px var(--accent)}
.card.dim{opacity:.35}
.card img{flex:1 1 0;min-height:0;width:100%;object-fit:contain;background:rgba(127,127,127,.08);pointer-events:none}
.body{flex:0 0 auto;max-height:60%;overflow:hidden;padding:12px 14px}
.card.noimg .body{flex:1 1 auto;max-height:none;display:flex;flex-direction:column;justify-content:center}
.title{font-size:18px;font-weight:650;margin:0 0 6px;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.card.noimg .title{font-size:21px}
dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:3px 10px;font-size:14px}
dt{color:var(--muted)}
dd{margin:0;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;word-break:break-word}
.controls{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;flex:0 0 auto}
.controls button{font:inherit;font-weight:600;font-size:15px;padding:12px 0;border-radius:12px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
.controls button:active{background:var(--line)}
#tie{color:var(--accent)}
.empty{margin:auto;color:var(--muted);text-align:center;padding:24px}
#ranking{flex:1;min-height:0;overflow:auto;padding:6px 16px 16px;-webkit-overflow-scrolling:touch}
.rhead{display:flex;gap:10px;position:sticky;top:0;background:var(--bg);padding-bottom:8px;z-index:1}
.rhead input{flex:1;min-width:0;font:inherit;padding:9px 12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
.rhead a{align-self:center;color:var(--accent);font-weight:600;text-decoration:none}
#list{list-style:none;margin:0;padding:0;background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}
#list li{display:flex;gap:12px;align-items:center;padding:10px 14px;border-top:1px solid var(--line)}
#list li:first-child{border-top:0}
.rk{min-width:2ch;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
.th{width:40px;height:40px;border-radius:8px;object-fit:cover;flex:0 0 auto;background:var(--line)}
.nm{flex:1;min-width:0}
.nm b{display:block;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.nm small{color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums}
#toast{position:fixed;left:50%;bottom:calc(84px + env(safe-area-inset-bottom));transform:translateX(-50%);
  background:var(--ink);color:var(--bg);padding:8px 14px;border-radius:999px;font-size:14px;font-weight:600;z-index:5}
@media (min-width:760px) and (min-aspect-ratio:1/1){
  #compare{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr auto;max-width:1100px;width:100%;margin:0 auto}
  .controls{grid-column:1/-1;justify-self:center;width:100%;max-width:420px}
  #ranking{max-width:760px;width:100%;margin:0 auto}
}
</style>
</head>
<body>
<header>
  <div class="tabs"><button id="tCompare" class="on">Compare</button><button id="tRank">Ranking</button></div>
  <div class="stat" id="stat"></div>
</header>
<main>
  <section id="compare">
    <div class="card" id="a"></div>
    <div class="card" id="b"></div>
    <div class="controls"><button id="undo">Undo</button><button id="tie">Equal</button><button id="skip">Skip</button></div>
    <div class="empty" id="empty" hidden>You need at least two entries to compare.</div>
  </section>
  <section id="ranking" hidden>
    <div class="rhead"><input id="q" type="search" placeholder="Filter" autocomplete="off"><a href="/api/ranking.csv" download="ranking.csv">CSV</a></div>
    <ol id="list"></ol>
  </section>
</main>
<div id="toast" hidden></div>
<script>
const $ = s => document.querySelector(s);
let pair = null, busy = false, rankingData = [], toastTimer;

async function api(path, body) {
  const opt = body === undefined ? {} :
    {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}
function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.hidden = true, 1400);
}
function setStats(s) { if (s) $('#stat').textContent = `${s.comparisons} compared · ${s.seen}/${s.total} seen`; }
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
function renderCard(node, item) {
  node.replaceChildren();
  node.className = 'card' + (item.image ? '' : ' noimg');
  if (item.image) { const img = el('img'); img.src = item.image; img.alt = ''; node.append(img); }
  const body = el('div', 'body');
  body.append(el('div', 'title', item.title));
  if (item.fields.length) {
    const dl = el('dl');
    for (const [k, v] of item.fields) dl.append(el('dt', null, k), el('dd', null, v));
    body.append(dl);
  }
  node.append(body);
}
function resetCards() { for (const c of [$('#a'), $('#b')]) c.classList.remove('picked', 'dim'); }
function show(d) {
  setStats(d.stats);
  pair = d.pair;
  const has = !!pair;
  $('#a').hidden = $('#b').hidden = $('.controls').hidden = !has;
  $('#empty').hidden = has;
  if (has) { renderCard($('#a'), pair[0]); renderCard($('#b'), pair[1]); }
}
async function vote(result, picked) {
  if (busy || !pair) return;
  busy = true;
  if (picked) { picked.classList.add('picked'); (picked.id === 'a' ? $('#b') : $('#a')).classList.add('dim'); }
  try {
    const [d] = await Promise.all([
      api('/api/vote', {a: pair[0].id, b: pair[1].id, result}),
      new Promise(r => setTimeout(r, 160)),
    ]);
    show(d);
  } catch (e) { resetCards(); toast('Error: ' + e.message); }
  finally { busy = false; }
}
async function skip() {
  if (busy) return; busy = true;
  try { show(await api('/api/pair')); } catch (e) { toast('Error: ' + e.message); }
  finally { busy = false; }
}
async function undo() {
  if (busy) return; busy = true;
  try {
    const d = await api('/api/undo', {});
    if (!d.undone) toast('Nothing to undo');
    else { if (d.pair) show(d); else setStats(d.stats); toast('Undone'); }
  } catch (e) { toast('Error: ' + e.message); }
  finally { busy = false; }
}
function tab(which) {
  const r = which === 'rank';
  $('#tCompare').classList.toggle('on', !r); $('#tRank').classList.toggle('on', r);
  $('#compare').hidden = r; $('#ranking').hidden = !r;
  if (r) loadRanking();
}
async function loadRanking() {
  try { const d = await api('/api/ranking'); setStats(d.stats); rankingData = d.items; renderRanking(); }
  catch (e) { toast('Error: ' + e.message); }
}
function renderRanking() {
  const q = $('#q').value.trim().toLowerCase();
  const rows = q ? rankingData.filter(x => x.title.toLowerCase().includes(q) || x.id.toLowerCase().includes(q)) : rankingData;
  const list = $('#list'); list.replaceChildren();
  for (const x of rows.slice(0, 1500)) {
    const li = el('li');
    li.append(el('span', 'rk', x.rank));
    if (x.image) { const img = el('img', 'th'); img.src = x.image; img.alt = ''; img.loading = 'lazy'; li.append(img); }
    const nm = el('div', 'nm');
    const rec = `${x.w}–${x.l}` + (x.t ? `–${x.t}` : '');
    nm.append(el('b', null, x.title), el('small', null, x.n ? `${x.rating} ±${x.pm} · ${rec}` : 'not compared yet'));
    li.append(nm);
    list.append(li);
  }
}
$('#a').addEventListener('click', () => vote(1, $('#a')));
$('#b').addEventListener('click', () => vote(0, $('#b')));
$('#tie').onclick = () => vote(0.5, null);
$('#skip').onclick = skip;
$('#undo').onclick = undo;
$('#tCompare').onclick = () => tab('compare');
$('#tRank').onclick = () => tab('rank');
$('#q').addEventListener('input', renderRanking);
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || !$('#ranking').hidden || e.metaKey || e.ctrlKey) return;
  const k = e.key;
  if (k === '1' || k === 'ArrowUp' || k === 'ArrowLeft') vote(1, $('#a'));
  else if (k === '2' || k === 'ArrowDown' || k === 'ArrowRight') vote(0, $('#b'));
  else if (k === 'e' || k === '=') vote(0.5, null);
  else if (k === 's' || k === ' ') { e.preventDefault(); skip(); }
  else if (k === 'z' || k === 'Backspace') undo();
});
skip();
</script>
</body>
</html>
"""


def make_icon(size=192):
    """Draw a simple app icon (two stacked cards) as a PNG, no libraries needed."""
    bg, top, bottom = (24, 27, 34), (111, 134, 255), (72, 77, 92)
    m, gap = size * 0.19, size * 0.06
    h = (size - 2 * m - gap) / 2
    r = size * 0.06
    raw = bytearray()
    for y in range(size):
        raw.append(0)
        for x in range(size):
            colour = bg
            for y0, c in ((m, top), (m + h + gap, bottom)):
                if m <= x < size - m and y0 <= y < y0 + h:
                    cx = min(max(x, m + r), size - m - r)
                    cy = min(max(y, y0 + r), y0 + h - r)
                    if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                        colour = c
            raw += bytes(colour)

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "Ranker"
    ranker = page = manifest = icon = None

    def log_message(self, fmt, *args):
        pass

    def send(self, code, body, ctype, cache="no-store", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code=200):
        self.send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        R = self.ranker
        if path in ("/", "/index.html"):
            self.send(200, self.page, "text/html; charset=utf-8")
        elif path == "/manifest.webmanifest":
            self.send(200, self.manifest, "application/manifest+json")
        elif path in ("/icon.png", "/apple-touch-icon.png", "/favicon.ico"):
            self.send(200, self.icon, "image/png", cache="max-age=86400")
        elif path == "/api/pair":
            self.json({"pair": R.next_pair(), "stats": R.stats()})
        elif path == "/api/ranking":
            items = R.ranking()
            self.json({"items": items, "stats": R.stats()})
        elif path == "/api/ranking.csv":
            self.send(
                200,
                R.ranking_csv(),
                "text/csv; charset=utf-8",
                extra={"Content-Disposition": 'attachment; filename="ranking.csv"'},
            )
        elif path.startswith("/img/"):
            f = R.ds.local_images.get(unquote(path[5:]))
            if not f or not f.is_file():
                return self.send(404, "image not found", "text/plain")
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            self.send(200, f.read_bytes(), ctype, cache="max-age=86400")
        else:
            self.send(404, "not found", "text/plain")

    def do_POST(self):
        path = urlparse(self.path).path
        R = self.ranker
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self.send(400, "invalid JSON", "text/plain")
        if path == "/api/vote":
            try:
                R.vote(str(body.get("a")), str(body.get("b")), body.get("result"))
            except ValueError as e:
                return self.send(400, str(e), "text/plain")
            self.json({"pair": R.next_pair(), "stats": R.stats()})
        elif path == "/api/undo":
            undone, pair = R.undo()
            self.json({"undone": undone, "pair": pair, "stats": R.stats()})
        else:
            self.send(404, "not found", "text/plain")


def main():
    ap = argparse.ArgumentParser(
        description="Rank a dataset with pairwise choices from your phone."
    )
    ap.add_argument("data", help="CSV, TSV, JSON or JSONL file")
    ap.add_argument("--id", dest="id_field")
    ap.add_argument("--title", dest="title_field")
    ap.add_argument("--show")
    ap.add_argument("--image", dest="image_field")
    ap.add_argument("--db")
    ap.add_argument("--name")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    ds = Dataset(args.data, args.id_field, args.title_field, args.show, args.image_field)
    db_path = args.db or str(ds.path) + ".prefs.sqlite"
    ranker = Ranker(ds, db_path)
    name = args.name or ds.path.stem.replace("_", " ").replace("-", " ").strip().title() or "Ranker"

    Handler.ranker = ranker
    Handler.page = PAGE.replace("__TITLE__", html.escape(name)).replace(
        "__SHORT__", html.escape(name[:14])
    )
    Handler.manifest = json.dumps(
        {
            "name": name,
            "short_name": name[:14],
            "start_url": "/",
            "display": "standalone",
            "background_color": "#111318",
            "theme_color": "#111318",
            "icons": [{"src": "/icon.png", "sizes": "192x192", "type": "image/png"}],
        }
    )
    Handler.icon = make_icon()

    print(f"Loaded {len(ds.order)} entries from {ds.path.name}")
    print(
        f"  id: {ds.id_field or '(row number)'}   title: {ds.title_field}   "
        f"showing: {', '.join(ds.show_fields) or '-'}   image: {ds.image_field or '-'}"
    )
    if not ds.id_field:
        print(
            '  note: no "id" column, so entries are identified by row number. '
            "If you reorder or insert rows later, use --id with a unique column instead."
        )
    print(f"Choices stored in {db_path} ({ranker.total} so far)")
    print(f"Open http://{args.host}:{args.port}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
