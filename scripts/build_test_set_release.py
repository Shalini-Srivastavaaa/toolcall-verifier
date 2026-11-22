"""Build the public test set `release/test_set/` and its personal-data check (D-032).

    python3 scripts/build_test_set_release.py          # writes release/test_set/{test,test_synth}.jsonl, schemas and pii_check.txt

Input: data/derived/final/test.jsonl (363 real rows) and test_synth.jsonl (374 synthetic rows) (same SHA-256 as the Kaggle upload files, D-026).
Only the fields needed for evaluation are kept: id, domain, conversation (the turns), candidate_call, tool_schemas_text, label, error_class, error_class_name, origin, class7_reason,
class7_reason_source. Review notes, annotator notes, run keys, agent model names and file provenance are dropped.

Personal-data check (`pii_check.txt`): patterns for e-mail addresses, phone numbers, card-like digit runs (with a Luhn test), SSN-like numbers, IBANs, street addresses, dates of birth and
capitalized word pairs that look like names are searched in every row's text. A hit counts as "found in the tau2-bench records" when the same string occurs in the benchmark's own
databases, task definitions or policies (data/raw/tau2-bench/data/tau2/domains/); for BFCL-based rows the same search is done without that reference and the hits are listed (masked).
Values are masked in the report (first 2 and last 2 characters). It cannot prove that no real personal data exists, only that nothing matching these patterns lies outside the benchmark's records.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "derived" / "final"
OUT = ROOT / "release" / "test_set"
TAU2 = ROOT / "data" / "raw" / "tau2-bench" / "data" / "tau2" / "domains"
KEEP_CORE = ("domain", "label", "error_class", "error_class_name", "origin")


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def public_row(r, synthetic):
    return {
        "id": r["trajectory_id"], "domain": r["domain"], "conversation": r["turns"], "candidate_call": r["candidate_tool_call"], "tool_schemas_text": r.get("tool_schemas_text", ""),
        "label": r["label"], "error_class": r.get("error_class"), "error_class_name": r.get("error_class_name"), "origin": r.get("origin"),
        "class7_reason": r.get("class7_reason"), "class7_reason_source": r.get("class7_reason_source"), "synthetic": synthetic,
    }


def row_text(r):
    parts = []
    for t in r["conversation"]:
        parts.append(str(t.get("content", "")))
        if "arguments" in t:
            parts.append(json.dumps(t["arguments"], ensure_ascii=False))
    parts.append(json.dumps(r["candidate_call"], ensure_ascii=False))
    return "\n".join(parts)


# ----------------------------------------------------------------------------------------------- the check

RX = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"),
    "phone": re.compile(r"(?<![\w.])\+?\d{1,2}?[\s.-]?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"),
    "card-like": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    "ssn-like": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "street address": re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way|Place|Pl|Circle|Terrace)\b\.?"),
    "date of birth": re.compile(r"\b(?:19\d{2}|20[0-2]\d)-\d{2}-\d{2}\b"),
}
NAME_RX = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")
# capitalized pairs that are not names: sentence openers and product / policy vocabulary (the pairs that remain are listed in the report for a look)
NOT_NAMES = set("""Thank You|Thanks For|Hello There|Good Morning|Good Afternoon|Good Evening|How Can|Please Provide|Please Confirm|Basic Economy|Business Class|Credit Card|Gift Card|Travel Certificate|Travel Insurance
|Is There|Let Me|Could You|Can You|Would You|Do You|Will You|If You|When You|Yes Please|No Problem|Sure Thing|I Will|I Can|I Have|I Am|That Is|This Is|What Is|Which One|One Moment|Just A|Pending Items
|Customer Service|Data Roaming|Airplane Mode|Mobile Data|Wi Fi|Data Saver|Network Mode|Sim Card|Apn Settings|Voice Mail|Next Steps|Best Regards|Kind Regards""".replace("\n", "").split("|"))


def luhn(digits):
    s, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - (9 if d * 2 > 9 else 0)
        s += d
        alt = not alt
    return s % 10 == 0


def mask(v):
    v = str(v)
    return v if len(v) <= 4 else v[:2] + "…" * 1 + "*" * min(len(v) - 4, 8) + v[-2:]


def tau2_corpus():
    texts = []
    for dom in ("retail", "airline", "telecom"):
        for f in (TAU2 / dom).rglob("*"):
            if f.is_file() and f.suffix in (".json", ".toml", ".md") and "voice" not in f.name:
                texts.append(f.read_text(errors="ignore"))
    corpus = "\n".join(texts)
    return corpus, re.sub(r"[^0-9a-z]", "", corpus.lower())


def name_sets(corpus):
    firsts = set(re.findall(r'(?:"first_name"|first_name)\s*[:=]\s*"([^"]+)"', corpus))
    lasts = set(re.findall(r'(?:"last_name"|last_name)\s*[:=]\s*"([^"]+)"', corpus))
    for full in re.findall(r'(?:"name"|"full_name"|full_name|\bname)\s*[:=]\s*"([A-Z][a-z]+ [A-Z][a-z]+)"', corpus):
        a, b = full.split()
        firsts.add(a)
        lasts.add(b)
    words = set(re.findall(r"\b[A-Z][a-z]{2,}\b", corpus))
    return firsts, lasts, words


def check(rows, corpus, squashed, use_ref):
    hits = defaultdict(Counter)           # kind -> Counter(value)
    unmatched = defaultdict(Counter)
    for r in rows:
        txt = row_text(r)
        for kind, rx in RX.items():
            for m in rx.finditer(txt):
                v = m.group(0).strip()
                if kind == "card-like":
                    digits = re.sub(r"\D", "", v)
                    if not (13 <= len(digits) <= 19) or not luhn(digits):
                        continue
                    key = digits
                    found = key in re.sub(r"\D", "", corpus) if use_ref else None
                else:
                    if kind == "date of birth" and not re.search(r"(?i)birth|dob|born", txt[max(0, m.start() - 80): m.end() + 40]):
                        continue   # an ISO date that is not near the words birth / dob / born (a flight or order date)
                    key = v
                    found = (v in corpus or re.sub(r"[^0-9a-z]", "", v.lower()) in squashed) if use_ref else None
                hits[kind][key] += 1
                if use_ref and not found:
                    unmatched[kind][key] += 1
        for m in NAME_RX.finditer(txt):
            v = m.group(0)
            if v in NOT_NAMES:
                continue
            a, b = v.split()
            # explained: the pair itself, a first name and a last name of the benchmark's records, or two words that both occur in the benchmark's text as capitalized words
            found = (v in corpus or (a in FIRSTS and b in LASTS) or (a in WORDS and b in WORDS)) if use_ref else None
            hits["name-like pair"][v] += 1
            if use_ref and not found:
                unmatched["name-like pair"][v] += 1
    return hits, unmatched


FIRSTS = LASTS = WORDS = set()


def main():
    global FIRSTS, LASTS, WORDS
    OUT.mkdir(parents=True, exist_ok=True)
    real = [public_row(r, False) for r in load(SRC / "test.jsonl")]
    synth = [public_row(r, True) for r in load(SRC / "test_synth.jsonl")]
    for name, rows in (("test", real), ("test_synth", synth)):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert len(real) == 363 and len(synth) == 374
    # make the folder self-contained (it is also uploaded alone as a Hugging Face dataset): the data license and the license texts it refers to
    import shutil
    shutil.copy2(ROOT / "DATA_LICENSE.md", OUT / "DATA_LICENSE.md")
    (OUT / "licenses").mkdir(exist_ok=True)
    for f in ("CC-BY-4.0.txt", "tau2-bench-MIT.txt", "BFCL-Gorilla-Apache-2.0.txt"):
        shutil.copy2(ROOT / "licenses" / f, OUT / "licenses" / f)
    corpus, squashed = tau2_corpus()
    FIRSTS, LASTS, WORDS = name_sets(corpus)
    print(f"benchmark records: {len(FIRSTS)} first names, {len(LASTS)} last names, {len(WORDS)} capitalized words")
    lines = ["PERSONAL DATA CHECK of release/test_set (scripts/build_test_set_release.py). Values are masked (first 2 and last 2 characters).", ""]
    summary = {}
    for name, rows, use_ref in (("test (363 rows, tau2-bench)", real, True), ("test_synth, tau2-bench rows", [r for r in synth if r["domain"] != "bfcl"], True), ("test_synth, BFCL rows", [r for r in synth if r["domain"] == "bfcl"], False)):
        hits, unmatched = check(rows, corpus, squashed, use_ref)
        lines.append(f"== {name}: {len(rows)} rows ==" + ("" if use_ref else "   (no reference records for BFCL: all hits are listed)"))
        for kind in list(RX) + ["name-like pair"]:
            tot, dist = sum(hits[kind].values()), len(hits[kind])
            um = unmatched[kind]
            lines.append(f"  {kind:16s} {tot:5d} occurrences, {dist:4d} distinct" + (f"; NOT found in the tau2-bench records: {sum(um.values())} occurrences, {len(um)} distinct" if use_ref else ""))
            show = um if use_ref else hits[kind]
            if show and kind != "name-like pair":
                lines.append("      " + ", ".join(f"{mask(v)} (x{c})" for v, c in show.most_common(12)))
            elif show:
                lines.append("      " + ", ".join(f"{v} (x{c})" for v, c in show.most_common(60)))
        summary[name] = {k: (sum(hits[k].values()), sum(unmatched[k].values())) for k in hits}
        lines.append("")
    (OUT / "pii_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for p in (OUT / "test.jsonl", OUT / "test_synth.jsonl"):
        print(p.name, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
