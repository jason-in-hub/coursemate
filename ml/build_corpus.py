"""Chunk the raw course documents into data/chunks.jsonl.

Run (from ml/): .venv/bin/python build_corpus.py
Each chunk: id, text, source (human-readable label), section, n_words.
- Course project page: split into its top-level sections, then into windows of <= 200 words, about
  150 words each, with OVERLAP words shared by neighbours (window size evened out within a section).
  A section of <= 200 words stays one chunk, so short sections (e.g. Important Dates) are below 100 words.
- Canvas assignments: one chunk per item; due_at converted from UTC to HKT (UTC+8).
- Canvas front page: one chunk.
"""
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data" / "chunks.jsonl"
MAX_WORDS, TARGET, OVERLAP = 200, 150, 20
SECTIONS = ["Important Dates", "Group Formation", "Project Overview", "Final Report", "Submission",
            "Poster Presentation Session", "Video Trailer"]
HKT = timezone(timedelta(hours=8))


def split_sections(text):
    cur, out = None, {}
    for line in text.splitlines():
        s = line.strip()
        if s in SECTIONS and s not in out:
            cur = s
            out[cur] = []
            continue
        if cur and s:
            out[cur].append(s)
    return out


def windows(L):
    if L <= MAX_WORDS:
        return [(0, L)]
    n = math.ceil((L - OVERLAP) / (TARGET - OVERLAP))
    size = math.ceil((L + (n - 1) * OVERLAP) / n)
    spans, start = [], 0
    while True:
        end = min(L, start + size)
        spans.append((start, end))
        if end == L:
            return spans
        start += size - OVERLAP


def slug(s):
    return "-".join("".join(c if c.isalnum() else " " for c in s.lower()).split())


def hkt(utc):
    if utc is None:
        return "no due date on Canvas"
    d = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(HKT)
    return f"{d:%a} {d.day} {d:%b %Y}, {d:%H:%M} HKT"


def main():
    chunks = []
    page = (RAW / "course_project_page.txt").read_text(encoding="utf-8")
    for sec, lines in split_sections(page).items():
        words = " ".join(lines).split()
        for i, (a, b) in enumerate(windows(len(words))):
            chunks.append({"id": f"page:{slug(sec)}:{i}", "text": " ".join(words[a:b]),
                           "source": f"Course project page · {sec}", "section": sec, "n_words": b - a})
    for item in json.loads((RAW / "canvas_assignments.json").read_text(encoding="utf-8")):
        types = ", ".join(t.replace("_", " ") for t in item["submission_types"])
        text = (f"Canvas assignment: {item['name']}. Due: {hkt(item['due_at_utc'])}. "
                f"Canvas points: {item['points_possible']:g}. Submission type: {types}.")
        chunks.append({"id": f"canvas:{slug(item['name'])}", "text": text,
                       "source": f"Canvas Assignments · {item['name']}", "section": "Canvas Assignments",
                       "n_words": len(text.split())})
    front = (RAW / "canvas_front_page.txt").read_text(encoding="utf-8").split()
    chunks.append({"id": "canvas:front-page", "text": " ".join(front), "source": "Canvas front page",
                   "section": "Canvas front page", "n_words": len(front)})
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"
    with OUT.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    pn = [c["n_words"] for c in chunks if c["id"].startswith("page:")]
    print(f"wrote {len(chunks)} chunks -> data/chunks.jsonl | page chunks {len(pn)}, words {sorted(pn)} "
          f"| canvas items {sum(c['section'] == 'Canvas Assignments' for c in chunks)} | front page 1 "
          f"({len(front)} words) | overlap {OVERLAP} words")


if __name__ == "__main__":
    main()
