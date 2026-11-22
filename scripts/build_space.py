"""Build `release/space/` (the Gradio demo for a Hugging Face Space) (D-032). Nothing is uploaded.

    python3 scripts/build_space.py

Picks the 12 curated examples from the PUBLIC test set (release/test_set/test.jsonl) by fixed rules (below), attaches the human label and the probability that the v3b model gave on the
Kaggle evaluation run (results/final_v3b/test_predictions.jsonl), and copies predict.py and its data/ next to app.py. app.py and the Space README are hand-written files in release/space/.

Selection rule: for each slot, the row with the SMALLEST conversation (at least 1,500 characters of turns, so that there is something to read) among the rows that match the slot. The decision
used for "caught", "missed" and "false alarm" is the 10% false-alarm threshold (0.7314) on the saved probabilities. The mix is deliberate and NOT representative: 6 errors the model catches
(Classes 1, 3, 4, 5, 6, 7), 3 it misses (Classes 7, 1, 3), 2 correct calls it allows and 1 correct call it wrongly flags. The order is shuffled with a fixed seed so that the list does not reveal the answer.
"""

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "release" / "space"
THR = json.load(open(ROOT / "verifier" / "data" / "thresholds.json"))["operating_points"]["fa10"]["threshold"]

# (outcome, error_class, origin, domain or None, tool or None)
SLOTS = [
    ("caught", 4, "natural", "telecom", "get_customer_by_phone"),
    ("caught", 7, "natural", "telecom", "refuel_data"),
    ("caught", 3, "natural", "telecom", "send_payment_request"),
    ("caught", 1, "natural", "airline", "book_reservation"),
    ("caught", 5, "induced", "retail", "get_order_details"),
    ("caught", 6, "induced", "airline", "update_reservation_flights"),
    ("missed", 7, "natural", "telecom", "refuel_data"),
    ("missed", 1, "natural", "retail", "modify_pending_order_items"),
    ("missed", 3, "natural", "airline", "get_flight_status"),
    ("allowed", None, "natural", "retail", "get_order_details"),
    ("allowed", None, "natural", "airline", "get_reservation_details"),
    ("false alarm", None, "natural", "airline", "cancel_reservation"),
]
NAMES = {1: "wrong argument value", 2: "stale state", 3: "wrong tool", 4: "repeated call", 5: "failure to propagate", 6: "ignored error", 7: "policy violation"}


def size(r):
    return sum(len(str(t.get("content", ""))) + len(json.dumps(t.get("arguments", ""))) for t in r["conversation"])


def main():
    rows = [json.loads(l) for l in open(ROOT / "release" / "test_set" / "test.jsonl")]
    pred = {json.loads(l)["trajectory_id"]: json.loads(l)["p_incorrect"] for l in open(ROOT / "results" / "final_v3b" / "test_predictions.jsonl")}
    for r in rows:
        r["p"] = pred[r["id"]]
        r["size"] = size(r)
        if r["label"] == "incorrect":
            r["outcome"] = "caught" if r["p"] >= THR else "missed"
        else:
            r["outcome"] = "false alarm" if r["p"] >= THR else "allowed"
    chosen = []
    for outcome, cls, origin, dom, tool in SLOTS:
        pool = [r for r in rows if r["outcome"] == outcome and r["error_class"] == cls and r["origin"] == origin and r["domain"] == dom and r["candidate_call"]["name"] == tool and r["size"] >= 1500
                and r["id"] not in {c["id"] for c in chosen}]
        assert pool, (outcome, cls, origin, dom, tool)
        chosen.append(min(pool, key=lambda r: r["size"]))
    random.Random(32).shuffle(chosen)
    examples = []
    for i, r in enumerate(chosen, 1):
        examples.append({
            "title": f"Example {i} · {r['domain']} · {r['candidate_call']['name']}",
            "id": r["id"], "domain": r["domain"], "conversation": r["conversation"], "candidate_call": r["candidate_call"], "tool_schemas_text": r["tool_schemas_text"],
            "human": {"label": r["label"], "error_class": r["error_class"], "error_class_name": NAMES.get(r["error_class"]), "origin": r["origin"], "class7_reason": r["class7_reason"],
                      "class7_reason_source": r["class7_reason_source"]},
            "saved_probability": r["p"], "outcome_at_fa10": r["outcome"],
        })
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"threshold_fa10": THR, "examples": examples}, open(OUT / "examples.json", "w"), ensure_ascii=False, indent=1)
    shutil.copy2(ROOT / "verifier" / "predict.py", OUT / "predict.py")
    shutil.rmtree(OUT / "data", ignore_errors=True)
    shutil.copytree(ROOT / "verifier" / "data", OUT / "data")
    for f in ("LICENSE", "NOTICE"):
        shutil.copy2(ROOT / f, OUT / f)
    shutil.copy2(ROOT / "DATA_LICENSE.md", OUT / "DATA_LICENSE.md")
    (OUT / "licenses").mkdir(exist_ok=True)
    for f in ("CC-BY-4.0.txt", "tau2-bench-MIT.txt", "BFCL-Gorilla-Apache-2.0.txt"):
        shutil.copy2(ROOT / "licenses" / f, OUT / "licenses" / f)
    for i, e in enumerate(examples, 1):
        print(f"{i:2d} {e['outcome_at_fa10']:11s} p={e['saved_probability']:.3f} {e['domain']:8s} {e['candidate_call']['name']:28s} class {e['human']['error_class']} {e['human']['origin']}  ({sum(len(str(t.get('content',''))) for t in e['conversation'])} chars, {len(e['conversation'])} turns)")


if __name__ == "__main__":
    main()
