#!/usr/bin/env python3
"""Generate docs/images/architecture.svg, the overview diagram used in README.md.

Standard library only. Run from anywhere:

    python3 docs/make_architecture_diagram.py

The SVG renders directly on GitHub. Edit the content in `build()` when the
pipeline changes, then re-run.
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent / "images" / "architecture.svg"

W = 1200
MARGIN = 20
SANS = "Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "SFMono-Regular, Menlo, Consolas, DejaVu Sans Mono, monospace"

# Box kinds: (fill, stroke)
KIND = {
    "plain": ("#e4e4e4", "#9b9b9b"),   # plain Rust, no AI
    "model": ("#cfe0fb", "#6f9be0"),   # calls an external model provider
    "result": ("#fbd6a8", "#e09a52"),  # result you use
    "hub": ("#dfe8f6", "#7f9fcf"),
}
# Panel tints: (fill, stroke)
PANEL = {
    "slate": ("#eff3f8", "#c9d5e4"),
    "green": ("#edf5e5", "#bcd6a6"),
    "teal": ("#e7f1ee", "#b3cfc5"),
    "peach": ("#fbe9e3", "#e9bcaa"),
    "yellow": ("#fdf5da", "#e6cf7c"),
    "orange": ("#fdebd7", "#ecc08c"),
    "lavender": ("#efedfb", "#c9c3ea"),
    "grey": ("#f1f1f1", "#d0d0d0"),
}
INK = "#222222"
MUTED = "#6b6b6b"
BLUE_INK = "#3567c9"
ARROW = "#555555"
GOOD = "#2f8a3a"


def text_width(s: str, size: float, mono: bool = False) -> float:
    return len(s) * size * (0.61 if mono else 0.53)


def wrap(s: str, max_px: float, size: float) -> list[str]:
    lines: list[str] = []
    for para in s.split("\n"):
        words, line = para.split(), ""
        for w in words:
            cand = f"{line} {w}".strip()
            if text_width(cand, size) <= max_px or not line:
                line = cand
            else:
                lines.append(line)
                line = w
        lines.append(line)
    return lines


class Svg:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    # ------------------------------------------------------------ primitives
    def text(self, x, y, s, size=11, *, color=INK, weight="normal", anchor="start",
             mono=False, italic=False):
        style = ' font-style="italic"' if italic else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{MONO if mono else SANS}" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}"{style}>{escape(s)}</text>'
        )

    def rect(self, x, y, w, h, fill, stroke, *, r=8, shadow=False, sw=1.2):
        if shadow:
            self.add(f'<rect x="{x + 2.5:.1f}" y="{y + 3:.1f}" width="{w:.1f}" height="{h:.1f}" '
                     f'rx="{r}" fill="#000" fill-opacity="0.13"/>')
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def arrow(self, pts, *, color=ARROW, label=None, label_at=None, both=False, width=1.4):
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        marker = "url(#ah-good)" if color == GOOD else "url(#ah)"
        start = f' marker-start="{marker}"' if both else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'
                 f'{start} marker-end="{marker}"/>')
        if label:
            lx, ly = label_at or pts[len(pts) // 2]
            self.text(lx, ly, label, 8.5, color=color if color == GOOD else MUTED, anchor="middle")

    # ------------------------------------------------------------ composites
    def panel(self, x, y, w, h, tint, label=None, label_size=11.5):
        fill, stroke = PANEL[tint]
        self.rect(x, y, w, h, fill, stroke, r=12)
        if label:
            self.text(x + w / 2, y + 20, label, label_size, anchor="middle")

    def chip(self, x, y, w, h, title, sub=None, kind="plain", size=10):
        fill, stroke = KIND[kind]
        self.rect(x, y, w, h, fill, stroke, r=6, shadow=True)
        cy = y + h / 2 + (-1 if sub else 3.5)
        self.text(x + w / 2, cy, title, size, anchor="middle", mono=True)
        if sub:
            self.text(x + w / 2, cy + 13, sub, 8, color=MUTED, anchor="middle")

    def card(self, x, y, w, h, title, tag, body, kind="plain"):
        fill, stroke = KIND[kind]
        self.rect(x, y, w, h, fill, stroke, r=8, shadow=True)
        self.text(x + 10, y + 20, title, 12, weight="bold")
        tag_color = BLUE_INK if kind == "model" else MUTED
        self.text(x + 10, y + 35, tag, 8.8, color=tag_color, mono=True)
        for i, line in enumerate(wrap(body, w - 20, 9.6)):
            self.text(x + 10, y + 54 + i * 13, line, 9.6)

    def chip_row(self, x, y, w, items, *, h=40, gap=14, kind="plain", size=10):
        n = len(items)
        cw = (w - gap * (n - 1)) / n
        for i, item in enumerate(items):
            title, sub = item if isinstance(item, tuple) else (item, None)
            self.chip(x + i * (cw + gap), y, cw, h, title, sub, kind, size)

    def section(self, y, number, title):
        self.text(MARGIN, y, f"{number}  ·  {title}", 15, weight="bold")
        self.add(f'<line x1="{MARGIN}" y1="{y - 24}" x2="{W - MARGIN}" y2="{y - 24}" '
                 f'stroke="#d6d6d6" stroke-width="1"/>')
        return y + 14

    def tree(self, x0, x1, y, root, children, *, kids_y_gap=46):
        """Root chip centred over evenly spaced child chips, joined by curves."""
        rw = text_width(root, 10, True) + 34
        rx = (x0 + x1) / 2 - rw / 2
        self.chip(rx, y, rw, 24, root, kind="hub", size=10)
        sizes = [text_width(c, 8.6, True) + 18 for c in children]
        gap = 8
        total = sum(sizes) + gap * (len(children) - 1)
        scale = min(1.0, (x1 - x0) / total)
        cx = (x0 + x1) / 2 - total * scale / 2
        cy = y + kids_y_gap
        for c, cw in zip(children, sizes):
            cw *= scale
            mid = cx + cw / 2
            sx, sy = (x0 + x1) / 2, y + 24
            self.add(f'<path d="M {sx:.1f} {sy:.1f} C {sx:.1f} {sy + 16:.1f}, {mid:.1f} {cy - 16:.1f}, '
                     f'{mid:.1f} {cy:.1f}" fill="none" stroke="#8d8d8d" stroke-width="1"/>')
            self.rect(cx, cy, cw, 20, *KIND["plain"], r=5, shadow=True)
            self.text(mid, cy + 13.5, c, 8.6, anchor="middle", mono=True)
            cx += cw + gap * scale
        return cy + 20

    def render(self, height: int) -> str:
        defs = (
            '<defs>'
            f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{ARROW}"/></marker>'
            f'<marker id="ah-good" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{GOOD}"/></marker>'
            '</defs>'
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}">{defs}'
            f'<rect width="{W}" height="{height}" fill="#ffffff"/>'
            + "".join(self.parts) + "</svg>\n"
        )


def build() -> str:
    s = Svg()
    R = W - MARGIN  # right edge
    IW = W - 2 * MARGIN  # inner width

    # ---------------------------------------------------------------- header
    s.text(MARGIN, 34, "Sievrant: architecture, query pipeline and the coverage gate", 22, weight="bold")
    s.text(MARGIN, 54, "Documents in  ->  evidence with a calibrated coverage guarantee out   |   "
                       "shown for one tenant, one corpus, one query", 11, color=MUTED)
    lx = 790
    for label, kind in [("plain Rust, no AI", "plain"), ("model provider call", "model"),
                        ("result you use", "result")]:
        fill, stroke = KIND[kind]
        s.add(f'<ellipse cx="{lx}" cy="30" rx="8" ry="6" fill="{fill}" stroke="{stroke}"/>')
        s.text(lx + 13, 33.5, label, 9.5)
        lx += 135

    # ---------------------------------------------------------------- 1
    y = s.section(100, 1, "Software architecture: who calls whom")
    s.panel(MARGIN, y, IW, 230, "slate")
    top = y + 20

    s.panel(40, top, 300, 190, "green", "Inputs")
    s.chip(55, top + 38, 130, 58, "your app", "Python · Node · REST · gRPC")
    s.chip(195, top + 38, 130, 58, ".env", "API keys · provider keys")
    s.chip(55, top + 112, 130, 58, "documents", "id · content · metadata · ACL")
    s.chip(195, top + 112, 130, 58, "labelled queries", "for calibration")

    hub_x, hub_w = 375, 265
    s.rect(hub_x, top + 8, hub_w, 112, *KIND["hub"], r=8, shadow=True)
    s.text(hub_x + hub_w / 2, top + 36, "SIEVRANT SERVER", 15, weight="bold", anchor="middle")
    s.text(hub_x + hub_w / 2, top + 54, "main.rs -> api/rest.rs -> sievrant::Corpus", 8.8,
           color=MUTED, anchor="middle", mono=True)
    s.text(hub_x + hub_w / 2, top + 74, "API key -> tenant · ACL before scoring", 9.6, anchor="middle")
    s.text(hub_x + hub_w / 2, top + 89, "one Corpus per (tenant, corpus)", 9.6, anchor="middle")
    s.text(hub_x + hub_w / 2, top + 104, "ingest · retrieve · calibrate · feedback", 9.6, anchor="middle")
    s.arrow([(340, top + 64), (hub_x, top + 64)])

    prov_x, prov_w = 778, 176
    s.rect(prov_x, top + 8, prov_w, 112, *KIND["model"], r=8, shadow=True)
    s.text(prov_x + prov_w / 2, top + 36, "MODEL PROVIDERS", 15, weight="bold", anchor="middle")
    s.text(prov_x + prov_w / 2, top + 54, "embeddings · reranker · LLM", 8.8, color=BLUE_INK,
           anchor="middle", mono=True)
    for i, line in enumerate(["OpenAI · Anthropic · Gemini", "Cohere · Voyage · Jina · Ollama",
                              "or offline 'local' (tests)"]):
        s.text(prov_x + prov_w / 2, top + 74 + i * 15, line, 9.6, anchor="middle")
    s.arrow([(hub_x + hub_w, top + 44), (prov_x, top + 44)], label="text · query · candidates",
            label_at=((hub_x + hub_w + prov_x) / 2, top + 38))
    s.arrow([(prov_x, top + 86), (hub_x + hub_w, top + 86)], label="vectors · scores · answer",
            label_at=((hub_x + hub_w + prov_x) / 2, top + 100))

    s.chip(hub_x, top + 148, 130, 44, "STORAGE", "data/tenants/<t>/<c>/", size=11)
    s.chip(hub_x + 150, top + 148, 130, 44, "QDRANT", "optional vector store", size=11)
    s.arrow([(hub_x + 65, top + 120), (hub_x + 65, top + 148)], both=True)
    s.arrow([(hub_x + 215, top + 120), (hub_x + 215, top + 148)], both=True)

    s.panel(970, top, 190, 190, "yellow", "Every request")
    for i, item in enumerate(["Bearer key -> tenant", "<= 256 in flight, else 503",
                              "timeout 120 s", "body <= 32 MB", "JSON errors + request id"]):
        s.chip(985, top + 32 + i * 31, 160, 24, item, size=9)

    # ---------------------------------------------------------------- 2
    y = s.section(y + 230 + 46, 2, "Query pipeline: when each model is called")
    s.panel(MARGIN, y, IW, 180, "teal", "Phase 1  ·  every query   (stage: retrieve)")
    cw, cg, cy = 200, 30, y + 34
    cards = [
        ("1  ROUTE", "sievrant-query · router",
         "Picks FAST / STANDARD / SLOW from the query text. Clients cannot force a route, "
         "so calibration stays valid. No model call.", "plain"),
        ("2  EMBED QUERY", "embedding provider",
         "in: the query, plus expansions if enabled. out: query vector. 1 call per query.", "model"),
        ("3  SEARCH", "sievrant-index",
         "Vector + BM25 in parallel. ACL filter applied before scoring. Tree and graph "
         "indexes are experimental, off by default. No model call.", "plain"),
        ("4  FUSE", "sievrant-math · rrf",
         "Reciprocal Rank Fusion, k = 60. Rank based, so scores from different indexes "
         "need no calibration. No model call.", "plain"),
        ("5  RERANK", "reranker provider",
         "in: query + top-N fused candidates. out: a relevance score per candidate. "
         "1 call per query.", "model"),
    ]
    for i, (t, tag, body, kind) in enumerate(cards):
        x = 40 + i * (cw + cg)
        s.card(x, cy, cw, 130, t, tag, body, kind)
        if i:
            s.arrow([(x - cg, cy + 65), (x, cy + 65)])

    y2 = y + 180 + 46
    s.arrow([(160, y + 180), (160, y2)], label="labelled queries run the same pipeline",
            label_at=(262, y + 200))
    s.arrow([(640, y + 180), (640, y2)], label="scored candidates of this query",
            label_at=(724, y + 200))

    s.panel(MARGIN, y2, 285, 205, "green", "Phase 2  ·  once per corpus")
    s.card(38, y2 + 34, 250, 155, "6  CALIBRATE", "calibrate.rs · conformal.rs",
           "Score each labelled query by its best relevant candidate. Threshold tau = the "
           "ceil((n+1) x coverage)-th smallest score. Saved with the pipeline fingerprint "
           "and corpus size. No model call (bootstrap mode asks the LLM to write questions).")

    s.panel(325, y2, R - 325, 205, "peach", "Phase 3  ·  every query   (stage: decide)")
    xs = [345, 552, 759, 966]
    w3 = 194
    s.card(xs[0], y2 + 34, w3, 130, "7  CONFORMAL GATE", "sievrant-math · conformal",
           "Keep candidates with 1 - score <= tau. Empty set -> ABSTAINED. Uncalibrated or "
           "stale -> no guarantee claimed, evidence still returned.")
    s.card(xs[1], y2 + 34, w3, 130, "8  SELECT", "submodular greedy",
           "Pick diverse evidence that fits the token budget, (1 - 1/e) approximation. "
           "Flags budget_truncated. No model call.")
    s.card(xs[2], y2 + 34, w3, 130, "9  ANSWER", "LLM provider",
           "in: query + selected evidence. out: answer text. 1 call, only when the gate "
           "passed and an LLM is configured.", "model")
    s.card(xs[3], y2 + 34, w3, 130, "10  EVIDENCE PACKET", "/v1/query · /v1/query/full",
           "decision, answer, evidence, guarantee, audit block, signed receipt. Logged so "
           "feedback can measure real coverage.", "result")
    for a, b in zip(xs, xs[1:]):
        s.arrow([(a + w3, y2 + 99), (b, y2 + 99)])
    s.arrow([(xs[1] + w3 / 2, y2 + 164), (xs[1] + w3 / 2, y2 + 186), (xs[3] + w3 / 2, y2 + 186),
             (xs[3] + w3 / 2, y2 + 164)], color=GOOD,
            label="abstained, or LLM = none: evidence only", label_at=(xs[2] + w3 / 2, y2 + 199))
    s.arrow([(288, y2 + 112), (xs[0], y2 + 112)], label="tau", label_at=(316, y2 + 106))

    # ---------------------------------------------------------------- 3
    y = s.section(y2 + 205 + 46, 3, "Model calls at a glance")
    pw = (IW - 40) / 3
    groups = [
        ("lavender", "embedding provider  ·  OpenAI / Gemini / Voyage / local", "model",
         [("embed chunks", "at ingest, batched"), ("embed query", "1 per query")]),
        ("slate", "reranker + LLM  ·  most of the cost", "model",
         [("rerank", "1 per query, top-N"), ("answer", "0 or 1 per query"),
          ("expand query", "optional, fast LLM"), ("bootstrap", "calibration questions")]),
        ("grey", "plain Rust  ·  no AI", "plain",
         [("router", "sievrant-query"), ("BM25 + RRF", "index · math"),
          ("conformal gate", "math · calibration"), ("audit + receipt", "audit · ed25519")]),
    ]
    for i, (tint, label, kind, items) in enumerate(groups):
        x = MARGIN + i * (pw + 20)
        s.panel(x, y, pw, 128, tint, label, 11)
        rows = [items[:2], items[2:]] if len(items) > 2 else [items]
        for r, row in enumerate(rows):
            s.chip_row(x + 16, y + 34 + r * 46, pw - 32, row, h=38, kind=kind)

    y += 128 + 16
    s.panel(MARGIN, y, IW, 82, "yellow", "What decides whether the guarantee applies")
    s.chip_row(36, y + 32, IW - 32, [
        ("calibration set", "labeled · synthetic · feedback"),
        ("pipeline fingerprint", "same models + settings"),
        ("corpus size", "within 1.5x of calibration"),
        ("requested coverage", "needs n >= c / (1 - c)"),
        ("observed coverage", "feedback · 95% Wilson interval")], h=38)
    y += 82 + 16
    s.panel(MARGIN, y, IW, 82, "orange", "Typical load")
    s.chip_row(36, y + 32, IW - 32, [
        ("one query", "1 embed · 1 rerank · 0-1 LLM"),
        ("ingest", "embedding calls per batch of chunks"),
        ("calibration", "one pipeline run per labelled query (200+ advised)")], h=38, kind="result")

    # ---------------------------------------------------------------- 4
    y = s.section(y + 82 + 46, 4, "Where data lives and what comes back")
    ph = 196
    s.panel(MARGIN, y, 470, ph, "slate", "data/tenants/<tenant>/<corpus>/   ·   per corpus")
    files = [("corpus.json", "name · created_at"), ("chunks.jsonl", "text + metadata (source of truth)"),
             ("vectors.bin", "memory backend"), ("bm25/", "Tantivy index"),
             ("calibration.json", "scores · fingerprint"), ("querylog.jsonl", "served queries"),
             ("feedback.jsonl", "labels -> observed coverage"), ("audit/", "append-only retrieval log")]
    for i, (t, sub) in enumerate(files):
        s.chip(36 + (i % 2) * 222, y + 34 + (i // 2) * 39, 210, 33, t, sub)

    s.arrow([(490, y + ph / 2), (528, y + ph / 2)])
    s.panel(530, y, 250, ph, "green", "Qdrant  ·  optional")
    s.chip(546, y + 40, 218, 46, "<prefix><tenant>__<corpus>", "one collection per corpus", size=9.5)
    s.text(655, y + 112, "SIEVRANT_VECTOR_BACKEND=qdrant", 8.8, color=MUTED, anchor="middle", mono=True)
    s.text(655, y + 128, "QDRANT_URL=http://qdrant:6334", 8.8, color=MUTED, anchor="middle", mono=True)
    s.text(655, y + 152, "text and metadata always come from", 9, color=MUTED, anchor="middle")
    s.text(655, y + 165, "chunks.jsonl, never from the index", 9, color=MUTED, anchor="middle")

    s.arrow([(780, y + ph / 2), (818, y + ph / 2)])
    s.panel(820, y, R - 820, ph, "yellow", "API responses")
    for i, (t, sub) in enumerate([("/v1/query", "summary + guarantee_status"),
                                  ("/v1/query/full", "full EvidencePacket"),
                                  ("/v1/corpora/{c}/calibration", "thresholds · holdout · drift"),
                                  ("/v1/corpora/{c}/gaps", "queries it abstained on")]):
        s.chip(836, y + 34 + i * 39, R - 820 - 32, 33, t, sub, "result", size=9.5)

    # ---------------------------------------------------------------- 5
    y = s.section(y + ph + 46, 5, "Data model of the Evidence Packet")
    s.panel(MARGIN, y, IW, 330, "green")
    yy = y + 18
    s.tree(40, R - 20, yy, "EvidencePacket",
           ["query_id", "decision", "answer", "coverage_guarantee", "guarantee", "evidence",
            "audit", "receipt", "route", "latency_ms"])
    yy += 92
    s.tree(40, 590, yy, "EvidenceChunk",
           ["chunk_id", "source_doc", "text", "rerank_score", "nonconformity_score"])
    s.tree(610, R - 20, yy, "GuaranteeInfo",
           ["status", "statement", "threshold", "calibration_set_id", "budget_truncated"])
    yy += 92
    s.tree(40, 440, yy, "AuditBlock",
           ["retrieval_stages_ms", "pipeline_fingerprint", "reranker", "warnings"])
    s.tree(450, 820, yy, "ObservedCoverage",
           ["rate", "observations", "lower_95", "upper_95", "drift_alert"])
    s.tree(830, R - 20, yy, "Receipt", ["alg", "key_id", "payload", "signature"])
    s.text(W / 2, y + 316, "decision is ANSWERED, ABSTAINED or EVIDENCE_ONLY   ·   coverage_guarantee is null "
                           "unless guarantee.status = calibrated   ·   receipt.payload is the exact signed JSON",
           9, color=MUTED, anchor="middle")

    # ---------------------------------------------------------------- 6
    y = s.section(y + 330 + 46, 6, "Code modules  ·  17 crates")
    half = (IW - 20) / 2
    s.panel(MARGIN, y, half, 86, "slate", "Entry points")
    s.chip_row(36, y + 34, half - 32, [("sievrant-server", "REST + gRPC"), ("sievrant-cli", "command line"),
                                       ("sievrant", "Corpus API · Services")], h=38, gap=10)
    s.panel(MARGIN + half + 20, y, half, 86, "peach", "Model access + bindings")
    s.chip_row(MARGIN + half + 36, y + 34, half - 32,
               [("sievrant-llm", "LLM + embedding registry"), ("sievrant-python", "PyO3"),
                ("sievrant-node", "napi-rs")], h=38, gap=10)
    y += 86 + 16
    s.panel(MARGIN, y, IW, 86, "yellow", "Retrieval pipeline")
    s.chip_row(36, y + 34, IW - 32, [
        ("sievrant-orchestrator", "runs the stages"), ("sievrant-query", "router · expansion"),
        ("sievrant-index", "vector · BM25 · tree · graph"), ("sievrant-rerank", "reranker providers"),
        ("sievrant-math", "RRF · submodular · conformal")], h=38, gap=10)
    y += 86 + 16
    s.panel(MARGIN, y, IW, 86, "grey", "Support")
    s.chip_row(36, y + 34, IW - 32, [
        ("sievrant-core", "types · config · errors"), ("sievrant-calibration", "store · feedback · drift"),
        ("sievrant-audit", "log · signed receipts"), ("sievrant-acl", "ACL resolver"),
        ("sievrant-storage", "WAL · LSM"), ("sievrant-observability", "logging · metering")],
        h=38, gap=10, size=9.5)
    y += 86

    s.text(MARGIN, y + 26, "Generated by docs/make_architecture_diagram.py", 9, color=MUTED)
    return s.render(math.ceil(y + 40))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
