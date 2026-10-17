"""Check paper/refs.bib against Crossref: title match, year, volume, pages, journal; report DOIs.

Run:  python src/tools_check_refs.py  -> results/refcheck.json and a printed report.
"""
import difflib
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "refs.bib"
HEAD = {"User-Agent": "drc-inverse-paper-refcheck/1.0 (mailto:bisimwamwim@gmail.com)"}


def parse(text):
    out = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        kind, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        fields = {f.lower(): re.sub(r"\s+", " ", v).strip()
                  for f, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body)}
        out.append(dict(kind=kind, key=key, **fields))
    return out


def plain(s):
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s or "")
    return re.sub(r"[{}\\\"'`~^]", "", s).lower().strip()


def query(e):
    first = plain(e.get("author", "")).split(",")[0].split(" and ")[0]
    q = f"{plain(e.get('title', ''))} {first}"
    r = requests.get("https://api.crossref.org/works", headers=HEAD, timeout=30,
                     params={"query.bibliographic": q, "rows": 3})
    r.raise_for_status()
    best, score = None, 0.0
    for it in r.json()["message"]["items"]:
        t = plain((it.get("title") or [""])[0])
        s = difflib.SequenceMatcher(None, t, plain(e.get("title", ""))).ratio()
        if s > score:
            best, score = it, s
    return best, score


def main():
    entries = parse(BIB.read_text(encoding="utf-8"))
    report = []
    for e in entries:
        if e["kind"] not in ("article", "inproceedings", "incollection", "book", "techreport"):
            report.append(dict(key=e["key"], status="skipped (misc)"))
            continue
        try:
            it, s = query(e)
        except Exception as exc:  # noqa: BLE001
            report.append(dict(key=e["key"], status=f"error {exc}"))
            continue
        time.sleep(0.5)
        if it is None:
            report.append(dict(key=e["key"], status="not found"))
            continue
        cr = dict(title=(it.get("title") or [""])[0], doi=it.get("DOI"),
                  journal=(it.get("container-title") or [""])[0],
                  year=(it.get("issued", {}).get("date-parts") or [[None]])[0][0],
                  volume=it.get("volume"), issue=it.get("issue"), pages=it.get("page"),
                  first_author=(it.get("author") or [{}])[0].get("family"))
        issues = []
        if s < 0.85:
            issues.append(f"title similarity {s:.2f}")
        if cr["year"] and str(cr["year"]) != e.get("year"):
            issues.append(f"year bib {e.get('year')} vs {cr['year']}")
        if cr["volume"] and e.get("volume") and cr["volume"] != e.get("volume"):
            issues.append(f"volume bib {e.get('volume')} vs {cr['volume']}")
        if cr["pages"] and e.get("pages"):
            bp = e["pages"].replace("--", "-")
            if bp != cr["pages"]:
                issues.append(f"pages bib {bp} vs {cr['pages']}")
        report.append(dict(key=e["key"], similarity=round(s, 2), issues=issues, crossref=cr,
                           status="ok" if not issues else "check"))
    (ROOT / "results" / "refcheck.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for r in report:
        cr = r.get("crossref", {})
        print(f"{r['key']:28s} {r['status']:8s} {r.get('similarity', '')!s:5s} doi={cr.get('doi')} "
              f"{'; '.join(r.get('issues', []))}")


if __name__ == "__main__":
    main()
