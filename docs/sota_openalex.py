#!/usr/bin/env python3
"""Reproducible state-of-the-art snapshot for geostat-rs, backed by OpenAlex.

Queries the OpenAlex REST API (free, no key) for the methodological frontiers
relevant to the engine, and writes a versionable Markdown snapshot with, per
topic: total article count, recent-vs-prior volume and trend, and the top-cited
recent works (with DOIs) ready for the paper's related-work section.

Title-scoped search (`title.search`) is deliberately precise: counts are lower
bounds and a *relative* signal of activity, not the absolute size of a field.

Stdlib only. Run from the repo root:
    python3 docs/sota_openalex.py --mailto you@example.com
    python3 docs/sota_openalex.py --since 2021 --top 5 --out docs/sota_snapshot.md
"""

import argparse
import datetime as dt
import json
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.openalex.org/works"

# (label, title.search query) — the frontiers we benchmark the engine against.
TOPICS = [
    ("Vecchia approximate kriging", "vecchia approximation"),
    ("Nearest-neighbour GP (NNGP)", "nearest-neighbor gaussian process"),
    ("SPDE / GMRF spatial", "spde spatial"),
    ("Multiple-point statistics", "multiple-point statistics"),
    ("Regression kriging (ML hybrid)", "regression kriging"),
    ("Random-forest kriging", "random forest kriging"),
    ("Compositional-data kriging", "compositional kriging"),
    ("Generative geostatistical sim.", "generative geostatistical simulation"),
]


def get(filters, mailto, **params):
    """Single OpenAlex GET; returns parsed JSON (or raises)."""
    q = {"filter": filters, "mailto": mailto, **params}
    url = f"{API}?{urllib.parse.urlencode(q)}"
    req = urllib.request.Request(url, headers={"User-Agent": f"geostat-rs-sota ({mailto})"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def trend_arrow(recent, prior):
    if prior == 0:
        return "new" if recent > 0 else "-"
    r = recent / prior
    if r >= 1.25:
        return "up"
    if r <= 0.8:
        return "down"
    return "flat"


def first_author(work):
    auths = work.get("authorships") or []
    if not auths:
        return "?"
    name = auths[0].get("author", {}).get("display_name", "?")
    last = name.split()[-1] if name != "?" else "?"
    return f"{last} et al." if len(auths) > 1 else last


def collect(mailto, since, top):
    half = max(1, (dt.date.today().year - since + 1))
    prior_lo = since - half
    rows = []
    for label, query in TOPICS:
        base = f"title.search:{query},type:article"
        hist = get(base, mailto, group_by="publication_year")
        by_year = {int(g["key"]): g["count"] for g in hist.get("group_by", [])}
        total = hist.get("meta", {}).get("count", 0)
        recent = sum(c for y, c in by_year.items() if y >= since)
        prior = sum(c for y, c in by_year.items() if prior_lo <= y < since)
        time.sleep(0.15)
        works = get(
            f"{base},from_publication_date:{since}-01-01",
            mailto,
            sort="cited_by_count:desc",
            per_page=top,
        )
        tops = [
            {
                "title": w.get("title") or "(untitled)",
                "year": w.get("publication_year"),
                "cites": w.get("cited_by_count", 0),
                "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                "author": first_author(w),
                "venue": ((w.get("primary_location") or {}).get("source") or {}).get(
                    "display_name", "?"
                ),
            }
            for w in works.get("results", [])
        ]
        rows.append(
            {
                "label": label,
                "query": query,
                "total": total,
                "recent": recent,
                "prior": prior,
                "trend": trend_arrow(recent, prior),
                "by_year": dict(sorted(by_year.items())),
                "top": tops,
            }
        )
        time.sleep(0.15)
    return rows, prior_lo


def render(rows, since, prior_lo, mailto):
    today = dt.date.today().isoformat()
    out = []
    out.append("# State-of-the-art snapshot (OpenAlex)\n")
    out.append(
        f"_Generated {today} via `docs/sota_openalex.py` against the OpenAlex API._\n"
    )
    out.append(
        "Counts use title-scoped search (`title.search`): precise but a **lower "
        "bound** and a *relative* activity signal, not the absolute field size. "
        f"Recent window = {since}-present; prior = {prior_lo}-{since - 1}.\n"
    )
    out.append("## Frontier activity\n")
    out.append("| Frontier | Total | Recent | Prior | Trend | In geostat-rs? |")
    out.append("|---|--:|--:|--:|:--:|---|")
    have = {
        "Vecchia approximate kriging": "no (dense LU)",
        "Nearest-neighbour GP (NNGP)": "no",
        "SPDE / GMRF spatial": "no",
        "Multiple-point statistics": "partial (SGS/SIS)",
        "Regression kriging (ML hybrid)": "yes",
        "Random-forest kriging": "yes (RF+resid)",
        "Compositional-data kriging": "no",
        "Generative geostatistical sim.": "no (out of scope)",
    }
    for r in rows:
        out.append(
            f"| {r['label']} | {r['total']} | {r['recent']} | {r['prior']} | "
            f"{r['trend']} | {have.get(r['label'], '?')} |"
        )
    out.append("")
    out.append("## Top recent works per frontier\n")
    for r in rows:
        out.append(f"### {r['label']}  \n`title.search:{r['query']}`\n")
        if not r["top"]:
            out.append("_No works._\n")
            continue
        for w in r["top"]:
            doi = f" doi:[{w['doi']}](https://doi.org/{w['doi']})" if w["doi"] else ""
            out.append(
                f"- **{w['cites']}** cites — {w['author']} ({w['year']}), "
                f"*{w['title']}* — {w['venue']}.{doi}"
            )
        out.append("")
    out.append("---\n")
    out.append(f"_Polite-pool contact: {mailto}._\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mailto", default="gran.huja@gmail.com", help="OpenAlex polite-pool email")
    ap.add_argument("--since", type=int, default=2022, help="recent-window start year")
    ap.add_argument("--top", type=int, default=4, help="top works per frontier")
    ap.add_argument("--out", default="docs/sota_snapshot.md", help="output Markdown path")
    args = ap.parse_args()

    try:
        rows, prior_lo = collect(args.mailto, args.since, args.top)
    except urllib.error.URLError as e:
        sys.exit(f"OpenAlex request failed: {e}")
    md = render(rows, args.since, prior_lo, args.mailto)
    with open(args.out, "w") as f:
        f.write(md)
    print(f"wrote {args.out} ({len(rows)} frontiers)")


if __name__ == "__main__":
    main()
