"""Side-by-side of what each model scores on synthetic, validation and real rows (D-030), read from the two evaluation reports (no predictions needed):
docs/final_v3b_eval_report.txt (ModernBERT v3b) and docs/final_qwen_v3b_eval_report.txt (Qwen2.5-1.5B). Prints a table and writes results/final_qwen_v3b/synth_real_gap.txt.

    python3 eval/synth_real_gap.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = {"ModernBERT v3b": ROOT / "docs" / "final_v3b_eval_report.txt", "Qwen2.5-1.5B (v3b data)": ROOT / "docs" / "final_qwen_v3b_eval_report.txt"}


def parse(path):
    t = path.read_text()
    real = t[t.index("=== REAL TEST SET") if "=== REAL TEST SET" in t else t.index("REAL TEST SET"):t.index("SYNTHETIC TEST SET")]
    synth = t[t.index("SYNTHETIC TEST SET"):]
    auc = lambda s: re.search(r"AUROC (\d\.\d+)  95% CI (\d\.\d+)-(\d\.\d+)", s).groups()  # noqa: E731
    rec = lambda s, k: re.search(rf"recall at {k}% false-alarm rate: (\d\.\d+)", s).group(1)  # noqa: E731
    out = {"real": auc(real), "real_r5": rec(real, 5), "real_r10": rec(real, 10), "synth": auc(synth), "synth_r5": rec(synth, 5), "synth_r10": rec(synth, 10),
           "val": re.search(r"validation AUROC of this model (\d\.\d+)", t).group(1), "real_origin_val": re.search(r"REAL-ORIGIN VALIDATION AUROC: (\d\.\d+)  95% CI (\d\.\d+)-(\d\.\d+)", t).groups()}
    out["groups"] = {m.group(1).strip(): (m.group(2), m.group(3), m.group(4)) for m in re.finditer(r"^   (BFCL|CriticTool|real-origin: induced|real-origin: natural|tau2 domains: synthetic, state flips, gold)\s+rows\s+(\d+) \(.*?\)\s+AUROC (\d\.\d+)  95% CI (\d\.\d+-\d\.\d+)", t, re.M)}
    out["domains"] = {m.group(1): m.group(2) for m in re.finditer(r"^  (airline|retail|telecom)\s+n\s+\d+ .*?AUROC (\d\.\d+)", real, re.M)}
    return out


def main():
    R = {k: parse(p) for k, p in REPORTS.items()}
    names = list(R)
    lines = [f"{'':58s}" + "".join(f"{n:>38s}" for n in names)]

    def row(label, f):
        lines.append(f"{label:58s}" + "".join(f"{f(R[n]):>38s}" for n in names))

    row("REAL test set (363 rows, human-reviewed): AUROC (95% CI)", lambda r: f"{r['real'][0]} ({r['real'][1]}-{r['real'][2]})")
    row("   recall at 5% / 10% false alarms", lambda r: f"{r['real_r5']} / {r['real_r10']}")
    for d in ("airline", "retail", "telecom"):
        row(f"   {d} AUROC", lambda r, d=d: r["domains"][d])
    row("SYNTHETIC test (test_synth, 374 rows, Classes 2 and 6): AUROC", lambda r: f"{r['synth'][0]} ({r['synth'][1]}-{r['synth'][2]})")
    row("   recall at 5% / 10% false alarms", lambda r: f"{r['synth_r5']} / {r['synth_r10']}")
    row("VALIDATION (749 rows, same distribution as training): AUROC", lambda r: r["val"])
    for g in ("BFCL", "CriticTool", "tau2 domains: synthetic, state flips, gold", "real-origin: natural", "real-origin: induced"):
        row(f"   {g}", lambda r, g=g: f"{r['groups'][g][1]} (CI {r['groups'][g][2]}, n={r['groups'][g][0]})" if g in r["groups"] else "n/a")
    row("   real-origin rows together (92 rows, 20 errors)", lambda r: f"{r['real_origin_val'][0]} ({r['real_origin_val'][1]}-{r['real_origin_val'][2]})")
    a, b = R[names[0]], R[names[1]]
    lines.append("")
    lines.append(f"validation real-origin AUROC minus REAL test AUROC: ModernBERT {float(a['real_origin_val'][0]) - float(a['real'][0]):+.3f}, Qwen {float(b['real_origin_val'][0]) - float(b['real'][0]):+.3f}")
    lines.append(f"synthetic test AUROC minus REAL test AUROC:          ModernBERT {float(a['synth'][0]) - float(a['real'][0]):+.3f}, Qwen {float(b['synth'][0]) - float(b['real'][0]):+.3f}")
    text = "\n".join(lines)
    print(text)
    out = ROOT / "results" / "final_qwen_v3b"
    out.mkdir(parents=True, exist_ok=True)
    (out / "synth_real_gap.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
