"""Build the next training / validation / test sets from every available source (D-022).

    python data/build_final_dataset.py                    # dry run: build everything in memory, print the would-be counts, write nothing
    python data/build_final_dataset.py --decisions FILE   # use another decisions file (for example a made-up one to exercise the code)
    python data/build_final_dataset.py --preview          # also writes the dry-run result to data/derived/final_preview/ (gitignored) to test notebooks and scripts
    python data/build_final_dataset.py --variant v3a --review-complete --write   # D-028: the v2 dataset + tool-balancing rows -> data/derived/final_v3a/ (v3b adds more real errors)
    python data/build_final_dataset.py --write --review-complete   # writes data/derived/final/*.jsonl; refuses without both flags

Sources
  current 7 files   train_pool, bfcl_pool, positives_class{2,4,6}_controls, negatives_pool_final (train side) and test_real (test side),
                    minus every row whose (domain, task_id) is a TEST-split task of data/derived/real_split.json (D-018: 1,309 rows)
  reviewed natural  candidates mined from failed tau2-bench runs (D-018)
  induced           candidates and paired correct handlings from the induced runs (D-019)
  CriticTool        data/derived/external_critictool.jsonl, training only
  state flips       data/derived/class7_stateflip.jsonl, training only (D-021)

Rules
  * TEST rows come only from human decisions (data/derived/review_queue_decisions.jsonl): test_real, plus accepted queue-1 and queue-4 candidates,
    queue-2 correct examples and queue-2 items the human marked wrong with a class (`found_in: "queue2"`, kept in test only). A queue-5 decision
    (the Class 7 / read-only re-review, docs/labeling_guide.md) supersedes the earlier decision of the same item (data/review_decisions.py); a
    queue-2 error that is in queue 5 and has no queue-5 decision yet is left out until it is re-reviewed. Test rows that are the same call (same run,
    tool and arguments) are kept once. Every row has `found_in`; every Class 7 row has `class7_reason` and `class7_reason_source`. No synthetic
    row is ever a test row (asserted).
  * TRAIN labels from the review pool are used only where they are trusted (D-022): a train-split candidate with auto_status auto_accepted is
    admitted only for the (origin, class) pairs in TRAIN_AUTO_ACCEPT: natural Classes 1, 4 and 7 and induced Class 5. Natural Classes 3, 5 and 6
    and induced Classes 2 and 6 are dropped unless a human reviewed the item (queue 3: an accept or a changed class admits it with the human's
    class, a reject drops it, whatever the class). The queue-1 precision behind the list is printed with the gate. Correct examples
    from the review pool are admitted only when a human accepted them (--train-positives accepted, the default) or, with --train-positives
    gated, when the human-confirmed share of that origin's queue-2 examples is at least 90% with at least 8 reviewed. Induced errors and
    their paired correct handlings enter together or not at all.
  * every row keeps `source` and `origin`; `split` is train | val | test | test_synth.
  * validation: whole (domain, task_id) groups chosen by a seeded search (see assign_val): about 10% of the training rows, at least 20
    airline rows and at least 15 rows of every error class that has enough rows; the result is stored in `split`. It no longer uses
    scikit-learn, so it does not depend on its version.
  * held-out synthetic test (Classes 2 and 6): whole (domain, task_id) groups that contain synthetic Class 2 / 6 negatives are moved out
    of training; their negatives and paired controls become `split: test_synth` (origin synthetic), so those two classes can also be
    reported on data that is not induced. The other rows of those groups are dropped, so nothing leaks.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from class7_reasons import suggest  # noqa: E402
from review_decisions import load_effective  # noqa: E402
DER = HERE / "derived"
OUT_DIR = DER / "final"
VARIANT_DIRS = {"v2": "final", "v3a": "final_v3a", "v3b": "final_v3b"}   # D-028: v3a = v2 + tool balancing from successful runs; v3b = v3a + errors from the newly localized train runs

SEED = 20261114
GATE_MIN_REVIEWED, GATE_PRECISION, POS_GATE = 8, 0.70, 0.90
# Which review-pool train labels are auto-accepted (D-022). Fixed by decision, from the queue-1 results (first labeler's confirmed share, >= 8 reviewed
# and >= 70% would pass exactly these four): natural class 1 11/11, natural class 4 8/11, natural class 7 29/41, induced class 5 19/19. Everything else
# (natural 3: 3/11, natural 5: 0/9, natural 6: 1/1 with 1 reviewed, induced 2: 3/16, induced 6: 5/13) is dropped unless a human reviews it.
# 2026-11-16: natural class 7 is 28/41 (68%) after the queue-5 rule (one read-only item became correct), just under 70%, and stays auto-accepted by decision:
# without it the training Class 7 rows are almost only retail state flips, while the real Class 7 test errors are mostly telecom and airline. About a third
# of those labels are not Class 7 (7 were class 1, 2 class 3, 4 not errors); see the D-022 update of that date. The other gates are unchanged.
TRAIN_AUTO_ACCEPT = {("natural", 1), ("natural", 4), ("natural", 7), ("induced", 5)}   # D-022; since D-026 the gate below is used instead of this set
# D-026 (2026-11-21): the real-data fix. The first final run trained on 16 real errors of 2,366. Natural Classes 1, 4 and 7 now enter training on the FIRST
# labeler's label alone (the second labeler's agreement is no longer required; human-measured precision of the first labeler 100%, 73%, 68%); natural 3, 5, 6
# stay out. Induced Class 5 (induced_propagation) enters when the auto-label accepted it (confirmed 19 of 19 on the test split). Each admitted error brings its
# pending positives (the earlier calls of the same run) as correct examples, minus those the rules mark as not correct (identical repeat, structural Class 7,
# a wrong line lookup).
TRAIN_FIRST_LABELER = {("natural", 1), ("natural", 4), ("natural", 7)}
TRAIN_INDUCED_AUTO = {("induced", 5)}
HELD_OUT = {2: 100, 6: 40}          # synthetic negatives to hold out per class (whole task groups, so about)
CLASS_NAMES = {1: "wrong_arg_value", 2: "stale_state", 3: "wrong_tool_selection", 4: "repeated_call", 5: "failure_to_propagate",
               6: "ignored_error", 7: "policy_violation"}
TOOL_RESULT_CAP = 2000              # as in data/prepare_tau2.py render_turns


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p) if line.strip()] if p.exists() else []


def tagged(rows, source=None, origin=None, split=None):
    out = []
    for r in rows:
        r = dict(r)
        if source is not None and not r.get("source"):
            r["source"] = source(r) if callable(source) else source
        if origin is not None and not r.get("origin"):
            r["origin"] = origin
        if split:
            r["split"] = split
        out.append(r)
    return out


def group_of(r: dict) -> str:
    return f"{r['domain']}::{r['task_id']}"


def dict_arguments(r: dict) -> dict:
    """Every tool call needs dict arguments (the notebooks' input formatters call .items()). Five CriticTool 'next step' rows have a
    free-text answer as `arguments` (a stringified set, the agent giving up); it is wrapped as {"text": ...} and the row is marked."""
    r = dict(r)
    c = dict(r["candidate_tool_call"])
    if not isinstance(c.get("arguments"), dict):
        c["arguments"] = {"text": str(c.get("arguments"))}
        r["candidate_tool_call"], r["arguments_wrapped"] = c, True
    for t in r["turns"]:
        if t.get("role") == "assistant_tool_call" and not isinstance(t.get("arguments"), dict):
            raise ValueError(f"{r['trajectory_id']}: a turn has non-dict arguments")
    return r


def class_of(r: dict):
    return int(r["error_class"]) if r["label"] == "incorrect" else None


# ───────────────────────────── 1. the current files ─────────────────────────────

def base_rows():
    d = lambda n: load(DER / f"{n}.jsonl")
    rows = (tagged(d("train_pool"), "tau2_gold", "natural") + tagged(d("bfcl_pool"), "bfcl_gold", "natural")
            + tagged(d("positives_class2_controls"), "synthetic_control_c2", "synthetic")
            + tagged(d("positives_class4_controls"), "synthetic_control_c4", "synthetic")
            + tagged(d("positives_class6_controls"), "synthetic_control_c6", "synthetic")
            + tagged(d("negatives_pool_final"), lambda r: f"synthetic_{r['corruption_method']}_c{r['error_class']}", "synthetic"))
    assert len(rows) == 7580, len(rows)
    return rows


def drop_new_test_tasks(rows):
    tasks = json.load(open(DER / "real_split.json"))["tasks"]
    test = {d: {str(t) for t in tasks[d]["test"]} for d in tasks}
    kept, dropped = [], Counter()
    for r in rows:
        if str(r["task_id"]) in test.get(r["domain"], ()):
            dropped[(r["domain"], r["source"])] += 1
        else:
            kept.append(r)
    return kept, dropped


# ───────────────────────────── 2. held-out synthetic Classes 2 / 6 ─────────────────────────────

def hold_out_synthetic(rows):
    by_group = defaultdict(list)
    for r in rows:
        by_group[group_of(r)].append(r)
    cand = defaultdict(Counter)
    for g, rs in by_group.items():
        for r in rs:
            if r["label"] == "incorrect" and r["origin"] == "synthetic" and r["error_class"] in HELD_OUT:
                cand[g][r["error_class"]] += 1
    order = sorted(cand)
    random.Random(SEED).shuffle(order)
    got, groups = Counter(), []
    for g in order:
        if all(got[c] >= n for c, n in HELD_OUT.items()):
            break
        if any(got[c] < HELD_OUT[c] and cand[g][c] for c in HELD_OUT):
            groups.append(g)
            got.update(cand[g])
    held, dropped, kept = [], [], []
    gset = set(groups)
    negs = {r["trajectory_id"] for g in groups for r in by_group[g] if r["label"] == "incorrect" and r["error_class"] in HELD_OUT and r["origin"] == "synthetic"}
    for r in rows:
        if group_of(r) not in gset:
            kept.append(r)
        elif (r["label"] == "incorrect" and r["trajectory_id"] in negs) or (r.get("paired_negative_id") in negs):
            held.append({**r, "split": "test_synth"})
        else:
            dropped.append(r)
    return kept, held, dropped, groups


# ───────────────────────────── 3. human decisions and the review pool ─────────────────────────────

def decisions(path: Path) -> dict:
    """One record per original item id with the queue-5 verdicts applied (data/review_decisions.py)."""
    return load_effective(path)


def review_pool():
    SRC = {"natural": ("real_candidates_review", "real_positive_candidates"), "induced": ("induced_candidates_review", "induced_positive_candidates")}
    out = {}
    for o, (c, p) in SRC.items():
        out[o] = {"cands": load(DER / f"{c}.jsonl"), "pos": load(DER / f"{p}.jsonl")}
    return out


def class_gate(items: dict, dec: dict) -> dict:
    """(origin, class) -> (reviewed, confirmed, admitted). `admitted` is the policy TRAIN_AUTO_ACCEPT; reviewed / confirmed are the human
    decisions on test-split error candidates (queues 1 and 4) shown beside it, and the data rule (>= 8 reviewed, >= 70% confirmed) is only reported."""
    stat = defaultdict(Counter)
    for o, pool in items.items():
        for c in pool["cands"]:
            d = dec.get(f"{o}::{c['candidate_id']}")
            if c["split"] != "test" or not d or d["queue"] not in (1, 4) or d["decision"] == "skip":
                continue
            k = (o, int(c["proposed_class"]))
            stat[k]["reviewed"] += 1
            stat[k]["confirmed"] += d["human_label"] != "reject" and d["human_class"] == int(c["proposed_class"])
    keys = set(stat) | TRAIN_AUTO_ACCEPT
    return {k: (stat[k]["reviewed"], stat[k]["confirmed"], k in TRAIN_AUTO_ACCEPT) for k in keys}


def positive_gate(items: dict, dec: dict) -> dict:
    out = {}
    for o, pool in items.items():
        n = ok = 0
        for p in pool["pos"]:
            d = dec.get(f"{o}::{p['positive_id']}")
            if d and d["queue"] == 2 and d["decision"] != "skip":
                n += 1
                ok += d["human_label"] == "correct"
        out[o] = (n, ok, n >= GATE_MIN_REVIEWED and ok / n >= POS_GATE)
    return out


class Runs:
    """Raw messages of the runs behind review rows, loaded only when a row has to be built."""

    def __init__(self):
        self.cache = None

    def preload(self, keys):
        """Natural runs in one pass over the results files (runs_by_file reads a whole file per call, so asking for them one by one is slow)."""
        from mine_real_errors import runs_by_file
        if self.cache is None:
            self.cache = {}
        for s in runs_by_file({k for k in keys if k not in self.cache}):
            self.cache[f"{s['file']}::{s['sim_id']}"] = s["messages"]

    def get(self, key: str, natural: bool):
        if self.cache is None:
            self.cache = {}
        if key not in self.cache:
            if natural:
                from mine_real_errors import runs_by_file
                for s in runs_by_file({key}):
                    self.cache[f"{s['file']}::{s['sim_id']}"] = s["messages"]
            else:
                for r in load(HERE / "raw" / "induced" / "induced_runs.jsonl"):
                    self.cache[r["run_id"]] = r["messages"]
        return self.cache[key]


def schema_text(domain: str) -> str:
    from tool_schemas import extract_tool_schemas, format_schemas_for_prompt
    return format_schemas_for_prompt(extract_tool_schemas(domain))


def history(messages: list[dict], call_index: int):
    """Turns before agent call `call_index`, in the training-row format (prepare_tau2.render_turns), plus the call itself."""
    from mine_real_errors import agent_calls
    from prepare_tau2 import render_turns
    calls = agent_calls(messages)
    c = calls[call_index]
    turns = render_turns(messages[: c["msg"]])
    for e in calls:  # earlier calls of the same assistant message come before this one
        if e["msg"] == c["msg"] and e["k"] < c["k"]:
            turns.append({"role": "assistant_tool_call", "name": e["name"], "arguments": e["arguments"]})
    for t in turns:
        if t["role"] == "tool":
            t["content"] = str(t["content"])[:TOOL_RESULT_CAP]
    return turns, {"name": c["name"], "arguments": c["arguments"]}


def review_row(o, kind, rec, label, error_class, split, runs, decision=None, tier="human_reviewed", found_in=None):
    """A full row for a candidate (kind error) or a correct example (kind correct)."""
    key = rec["run_key"]
    fc = rec["flagged_call"] if kind == "error" else {"call_index": rec["call_index"]}
    turns, call = history(runs.get(key, o == "natural"), fc["call_index"])
    rid = rec["candidate_id"] if kind == "error" else rec["positive_id"]
    r = {"trajectory_id": f"{o}::{rid}", "domain": rec["domain"], "task_id": str(rec["task_id"]), "source_model": rec.get("source_model"),
         "tool_schemas_text": schema_text(rec["domain"]), "turns": turns, "candidate_tool_call": call, "gold_action": None,
         "label": label, "label_tier": tier, "split": split,
         "error_class": error_class, "error_class_name": CLASS_NAMES.get(error_class) if error_class else None,
         "source": rec.get("source") or ("tau2_real_failed_run" if kind == "error" else "tau2_real_correct_call"), "origin": o}
    r["run_key"], r["call_index"] = key, fc["call_index"]
    if found_in:
        r["found_in"] = found_in
    if decision:
        r["review"] = {"item_id": decision["item_id"], "queue": decision["queue"], "decision": decision["decision"], "note": decision.get("human_note", "")}
        if decision.get("superseded_by"):
            r["review"]["superseded_by"], r["review"]["applied_by"] = decision["superseded_by"], decision.get("applied_by", "human")
        if decision.get("tag"):
            r["tags"] = [decision["tag"]]
    if error_class == 7:
        cr = (decision or {}).get("class7_reason")
        if cr:
            r["class7_reason"], r["class7_reason_source"] = cr, (decision or {}).get("class7_reason_source", "human")
    return r


def screen_positive(p, o, runs):
    """The rules that decide whether a pending positive (a call before the first flagged call) may be used as a correct example: an identical repeat of an earlier call
    (Class 4), a write the structural Class 7 pre-fill flags, and a line lookup the line-probe rule says is a wrong line (Class 1) are not correct. Everything else
    is kept: a read-only call is never Class 7 and a first line lookup is correct (labeling guide). Returns None when the call may be used, else the reason."""
    from class7_reasons import suggest
    from line_probe_rule import judge
    from mine_real_errors import render_full
    turns, call_turn = render_full(runs.get(p["run_key"], o == "natural"))
    t = call_turn[p["call_index"]]
    call = {"name": turns[t]["name"], "arguments": turns[t]["arguments"]}
    s = suggest(turns[:t], call, p["domain"])
    if s["verdict"] == "class4":
        return "identical repeat of an earlier call (Class 4)"
    if s["verdict"] == "class7":
        return f"structural Class 7 signal ({s['reason']})"
    if judge(turns[:t], call)["verdict"] == "class1":
        return "line lookup of a different line than the known target (Class 1)"
    return None


def screen_candidate(c, o, runs):
    """The labeling-guide rules applied to a first-labeler error candidate before it becomes a training error (D-026), so that training does not contradict the test set:
    a read-only call proposed as Class 7 is not an error (guide rule 1), and a first lookup of a customer's line (`get_details_by_id` on a line id) that the line-probe
    rule judges correct is not an error. Returns None when the candidate stays an error, else the reason."""
    from class7_reasons import is_read_only
    from line_probe_rule import judge
    from mine_real_errors import render_full
    turns, call_turn = render_full(runs.get(c["run_key"], o == "natural"))
    t = call_turn[c["flagged_call"]["call_index"]]
    call = {"name": turns[t]["name"], "arguments": turns[t]["arguments"]}
    if int(c["proposed_class"]) == 7 and is_read_only(call["name"]):
        return "a read-only call is never Class 7"
    if judge(turns[:t], call)["verdict"] == "correct":
        return "line lookup judged correct by the line-probe rule"
    return None


def reviewed_rows(items, dec, gate, pgate, train_positives, runs, q5_pending=frozenset(), screen_errors=True):
    test, train, notes = [], [], Counter()
    admitted = {}                                   # (origin, class) -> admitted train errors; D-026 report
    pos_stats, err_stats = Counter(), Counter()
    for o, pool in items.items():
        adm_cands, adm_runs = set(), set()
        for c in pool["cands"]:
            if c.get("batch") == "train_more":    # D-028: candidates of the newly localized train runs are handled after the validation split (late_candidate_rows)
                continue
            d = dec.get(f"{o}::{c['candidate_id']}")
            if c["split"] == "test":
                if d and d["queue"] in (1, 4) and d["decision"] != "skip":
                    if d["human_label"] == "reject":
                        notes["test candidates rejected by the reviewer"] += 1
                    else:
                        test.append(review_row(o, "error", c, "incorrect", int(d["human_class"]), "test", runs, d, found_in=f"queue{d['queue']}"))
                else:
                    notes["test candidates not reviewed yet (not in the test set)"] += 1
            else:
                k = (o, int(c["proposed_class"]))
                row = None
                if d and d["queue"] == 3 and d["decision"] != "skip":  # a human looked at this train candidate: the human decides, whatever the class gate says
                    if d["human_label"] == "reject":
                        notes[f"train candidates rejected by the reviewer ({o})"] += 1
                    else:
                        row = review_row(o, "error", c, "incorrect", int(d["human_class"]), "train", runs, d, tier="human_reviewed")
                elif k in TRAIN_FIRST_LABELER:     # D-026: the first labeler's label alone, whatever the second labeler said
                    adm_cands.add(c["candidate_id"])   # its pending positives are admitted even when the rules below say the flagged call is not an error
                    why = screen_candidate(c, o, runs) if screen_errors else None
                    if why:
                        err_stats[(k, why)] += 1
                    else:
                        row = review_row(o, "error", c, "incorrect", int(c["proposed_class"]), "train", runs, tier="first_labeler_gated")
                        row["second_labeler_agreed"] = c.get("auto_status") == "auto_accepted"
                elif k in TRAIN_INDUCED_AUTO and c.get("auto_status") == "auto_accepted":
                    row = review_row(o, "error", c, "incorrect", int(c["proposed_class"]), "train", runs, tier="auto_accepted_gated")
                else:
                    notes[f"train candidates held back ({o}, class {int(c['proposed_class'])}, {c.get('auto_status')})"] += 1
                if row is not None:
                    row["real_origin"] = True
                    train.append(row)
                    adm_cands.add(c["candidate_id"])
                    adm_runs.add(c["run_key"])
                    admitted[k] = admitted.get(k, 0) + 1
        for p in pool["pos"]:
            if p.get("batch") == "train_more":
                continue
            d = dec.get(f"{o}::{p['positive_id']}")
            if p["split"] == "test":
                if d and d["queue"] == 2 and d["decision"] != "skip":
                    if d["human_label"] == "correct":
                        test.append(review_row(o, "correct", p, "correct", None, "test", runs, d, found_in="queue2"))
                    elif d["human_class"] and f"{o}::{p['positive_id']}" in q5_pending:
                        notes["queue-2 errors waiting for the queue-5 re-review (left out until then)"] += 1
                    elif d["human_class"]:
                        row = review_row(o, "correct", p, "incorrect", int(d["human_class"]), "test", runs, d, found_in="queue2")
                        row["source"] = "queue2_found_error"   # a real agent call a human found wrong in the correct-example check, not an assumed-correct call
                        test.append(row)
                    else:
                        notes["queue-2 items judged wrong without a class (left out)"] += 1
                else:
                    notes["test correct examples not reviewed (not in the test set)"] += 1
            else:
                if train_positives == "gated" and pgate.get(o, (0, 0, False))[2]:
                    train.append(review_row(o, "correct", p, "correct", None, "train", runs, tier="auto_unreviewed_gated"))
                elif train_positives == "linked":  # D-026: the pending positives of an admitted error (same run, earlier calls), after the rules
                    linked = (p.get("linked_candidate_id") in adm_cands) if o == "natural" else (p["run_key"] in adm_runs)
                    if not linked:
                        pos_stats[(o, "not linked to an admitted error")] += 1
                    else:
                        why = screen_positive(p, o, runs)
                        if why:
                            pos_stats[(o, "excluded by the rules: " + why.split(" (")[0])] += 1
                        else:
                            row = review_row(o, "correct", p, "correct", None, "train", runs, tier="linked_positive_gated")
                            row["real_origin"] = True
                            train.append(row)
                            pos_stats[(o, "admitted")] += 1
                else:
                    notes[f"train correct examples held back ({o}): not human-accepted"] += 1
    if admitted:
        print("\nreal-origin training errors admitted (D-026):", {f"{o} class {c}": n for (o, c), n in sorted(admitted.items())})
    for ((o, cl), why), n in sorted(err_stats.items(), key=str):
        print(f"   first-labeler errors NOT used ({o} class {cl}): {n:3d}  {why}")
    for (o, what), n in sorted(pos_stats.items()):
        print(f"   correct examples ({o}): {n:4d}  {what}")
    return test, train, notes


# ───────────────────────────── 4. assemble, split, report ─────────────────────────────

VAL_FRACTION, VAL_MIN_AIRLINE, VAL_MIN_TELECOM, VAL_CLASS_ROWS, VAL_ENOUGH = 0.10, 20, 20, 15, 100
DOMAIN_WEIGHT = 30   # a missing domain row costs this much more than a missing class row: when the minimums cannot all be met, the domains win


def assign_val(train, iters=60000):
    """Validation = whole (domain, task_id) groups, found by a seeded search (simulated annealing over group assignments), so that
    validation is about 10% of the training rows, has at least 20 airline rows, at least 20 telecom rows and at least 15 rows of every error
    class that has enough rows (at least 100 in train + validation; a smaller class gets 10% of its rows). Python's random and numpy with a fixed
    seed only: the result does not depend on the scikit-learn version (D-020). If the minimums cannot all be met together, the domain
    minimums are weighted DOMAIN_WEIGHT/8 times more than the class minimums and the missed class targets are printed. Test rows are
    not touched."""
    import numpy as np
    gid = {}
    for r in train:
        gid.setdefault(group_of(r), len(gid))
    names = list(gid)
    F = ["rows", "airline", "incorrect"] + [f"c{k}" for k in range(1, 8)] + ["telecom"]
    V = np.zeros((len(names), len(F)))
    for r in train:
        v = V[gid[group_of(r)]]
        v[0] += 1
        v[1] += r["domain"] == "airline"
        v[10] += r["domain"] == "telecom"
        if r["label"] == "incorrect":
            v[2] += 1
            v[2 + int(r["error_class"])] += 1
    T = V.sum(0)
    tgt = np.array([VAL_FRACTION * T[0], VAL_MIN_AIRLINE, VAL_FRACTION * T[2]] + [VAL_CLASS_ROWS if T[2 + k] >= VAL_ENOUGH else VAL_FRACTION * T[2 + k] for k in range(1, 8)] + [VAL_MIN_TELECOM])
    CLS, DOM = list(range(3, 10)), [1, 10]

    def loss(v):
        size = max(0.0, abs(v[0] - tgt[0]) - 0.01 * T[0]) / T[0] * 60          # about 10% of the rows (a 1% band is free)
        short = sum(max(0.0, tgt[i] - v[i]) / max(tgt[i], 1) for i in CLS) + DOMAIN_WEIGHT / 8 * sum(max(0.0, tgt[i] - v[i]) / max(tgt[i], 1) for i in DOM)  # every class and the airline and telecom minimums; domains count more
        over = sum(max(0.0, v[i] - max(1.6 * tgt[i], 0.14 * T[i])) / max(T[i], 1) for i in CLS + DOM)  # but do not empty a class or a domain from training
        share = abs(v[2] / max(v[0], 1) - T[2] / T[0])                           # keep the incorrect share
        return size + 8 * short + 4 * over + 2 * share

    rng = random.Random(SEED)
    order = list(range(len(names)))
    rng.shuffle(order)
    inval = np.zeros(len(names), bool)
    cur = np.zeros(len(F))
    for g in order:                                                             # start: random groups until 10% of the rows
        if cur[0] >= tgt[0]:
            break
        inval[g] = True
        cur += V[g]
    cur_loss, best, best_loss = loss(cur), inval.copy(), loss(cur)
    for it in range(iters):
        temp = 0.5 * (1 - it / iters) + 1e-4
        g = rng.randrange(len(names))
        cand, nv = inval.copy(), cur.copy()
        for x in ([g] if rng.random() < 0.4 else [g, rng.randrange(len(names))]):   # toggle one group, or two at once (a swap when one is in and one out)
            cand[x] = ~cand[x]
            nv += V[x] if cand[x] else -V[x]
        nl = loss(nv)
        if nl <= cur_loss or rng.random() < np.exp((cur_loss - nl) / temp):
            inval, cur, cur_loss = cand, nv, nl
            if cur_loss < best_loss:
                best, best_loss = inval.copy(), cur_loss
    val_groups = {names[i] for i in np.flatnonzero(best)}
    v = V[best].sum(0)
    print(f"\nvalidation search (seeded, {iters} steps, loss {best_loss:.3f}): {len(val_groups)} of {len(names)} task groups in validation")
    print(f"   rows {int(v[0])} of {int(T[0])} ({v[0] / T[0]:.1%}); airline rows {int(v[1])} (target >= {VAL_MIN_AIRLINE}); telecom rows {int(v[10])} (target >= {VAL_MIN_TELECOM}); incorrect {int(v[2])} ({v[2] / v[0]:.1%} vs {T[2] / T[0]:.1%} overall)")
    missed = [f"class {k}" for k in range(1, 8) if v[2 + k] < tgt[2 + k] - 1e-9] + [f"{n} rows" for n, i in (("airline", 1), ("telecom", 10)) if v[i] < tgt[i]]
    print("   all minimums met" if not missed else f"   MISSED targets: {', '.join(missed)}")
    for k in range(1, 8):
        ok = "" if v[2 + k] >= tgt[2 + k] - 1e-9 else "   <-- below target"
        print(f"   class {k}: validation {int(v[2 + k]):4d} of {int(T[2 + k]):4d} (target >= {tgt[2 + k]:.0f}){ok}")
    return [{**r, "split": "val" if group_of(r) in val_groups else "train"} for r in train]


def table(rows, key, title):
    splits = ["train", "val", "test", "test_synth"]
    cnt = Counter((key(r), r["split"]) for r in rows)
    names = sorted({k for k, _ in cnt}, key=str)
    print(f"\n{title}")
    print(f"   {'':30s}" + "".join(f"{s:>11s}" for s in splits) + f"{'total':>9s}")
    for n in names:
        vals = [cnt[(n, s)] for s in splits]
        print(f"   {str(n):30s}" + "".join(f"{v:>11d}" for v in vals) + f"{sum(vals):>9d}")
    tot = [sum(cnt[(n, s)] for n in names) for s in splits]
    print(f"   {'total':30s}" + "".join(f"{v:>11d}" for v in tot) + f"{sum(tot):>9d}")


def apply_queue5_to_test_real(test_real, dec):
    """A later-layer decision (queue 5 on the 8 Class 7 rows, the line-probe rule on line-ID lookups) supersedes the phase-4 hand label of a test_real row."""
    out, changed = [], Counter()
    for r in test_real:
        d = dec.get(f"test_real::{r['trajectory_id']}")
        if not d or not d.get("superseded_by"):
            out.append(r)
            continue
        r = dict(r)
        r["review"] = {"item_id": d["item_id"], "queue": 5, "decision": d["decision"], "note": d.get("human_note", ""), "superseded_by": d["superseded_by"], "superseded_chain": d["superseded_chain"], "applied_by": d.get("applied_by", "human"), "earlier": d["earlier"]}
        if d["human_label"] == "correct":
            r.update(label="correct", error_class=None, error_class_name=None, label_tier="human_reviewed")
            changed["now correct"] += 1
        else:
            cls = int(d["human_class"])
            r.update(error_class=cls, error_class_name=CLASS_NAMES[cls], label_tier="human_reviewed")
            changed[f"class {cls}"] += 1
            if cls == 7 and d.get("class7_reason"):
                r["class7_reason"], r["class7_reason_source"] = d["class7_reason"], d.get("class7_reason_source", "human")
        if d.get("tag"):
            r["tags"] = [d["tag"]]
        out.append(r)
    if changed:
        print(f"\ntest_real rows re-judged by a later layer (queue 5 or the line-probe rule; the verdict replaces the phase-4 label): {dict(changed)}")
    return out


def call_keys(r):
    """Two keys for 'the same call': the exact one (run and call index, known for review rows) and the content one (run, tool, arguments; the only one
    test_real rows have). A content match alone is not a duplicate for two review rows: an agent that repeats a call (Class 4) repeats its content."""
    c = r["candidate_tool_call"]
    exact = (r["run_key"], r["call_index"]) if r.get("call_index") is not None else None
    return exact, (r["run_key"], c["name"], json.dumps(c["arguments"], sort_keys=True, default=str))


def dedupe_test(test_real, r_test):
    """Test rows that are the same call are kept once: test_real first, then queue 1 / 4 candidates, then queue 2. A call that is an error in one source
    and correct in another is kept as the error (a human found it wrong) and reported."""
    prio = {"test_real": 0, "queue1": 1, "queue4": 1, "queue2": 2}
    exact_seen, content_seen, out, dups = {}, {}, [], []
    for r in sorted(test_real + r_test, key=lambda r: (prio[r["found_in"]], r["label"] != "incorrect")):
        ex, ct = call_keys(r)
        kept = exact_seen.get(ex) if ex else None
        if kept is None and ct in content_seen and "test_real" in (r["found_in"], content_seen[ct]["found_in"]):
            kept = content_seen[ct]
        if kept is not None:
            dups.append((r, kept))
            continue
        if ex:
            exact_seen[ex] = r
        content_seen.setdefault(ct, r)
        out.append(r)
    if dups:
        print(f"\nduplicate test calls removed: {len(dups)}")
        for r, kept in dups:
            print(f"   {r['trajectory_id'][-60:]} ({r['found_in']}, {r['label']}, class {r.get('error_class')}) is the same call as {kept['trajectory_id'][-60:]} ({kept['found_in']}, {kept['label']}, class {kept.get('error_class')})"
                  + ("   <-- the two sources disagree" if (r["label"], r.get("error_class")) != (kept["label"], kept.get("error_class")) else ""))
    else:
        print("\nduplicate test calls: none (no test row is the same call as another: same run and call index; test_real rows also by run, tool and arguments)")
    keep = {id(r) for r in out}
    return [r for r in test_real if id(r) in keep], [r for r in r_test if id(r) in keep]


def content_key(r):
    import hashlib
    return hashlib.sha1(json.dumps([r["turns"], r["candidate_tool_call"]], sort_keys=True, default=str).encode()).hexdigest()


def dedupe_content(final):
    """Rows with identical content (the same turns and the same candidate call) are removed, never merged:
    (a) a train or validation row whose content is in test or test_synth is dropped (the human test label wins);
    (b) inside test and inside test_synth, identical rows are kept once (test_real first, then queue 1 / 4, then queue 2, then the smaller id);
    (c) a group whose labels conflict is NOT resolved: all its rows stay, and the group is listed so a human decides.
    Returns (rows, conflicts). Prints what was removed per class."""
    prio = {"test_real": 0, "queue1": 1, "queue4": 1, "queue2": 2}
    lab = lambda r: f"class {r['error_class']}" if r["label"] == "incorrect" else "correct"
    held = {content_key(r) for r in final if r["split"] in ("test", "test_synth")}
    drop, why = set(), {}
    for r in final:
        if r["split"] in ("train", "val") and content_key(r) in held:
            drop.add(id(r))
            why[id(r)] = "train/val row with the content of a test row"
    conflicts = []
    for sp in ("test", "test_synth"):
        by = defaultdict(list)
        for r in final:
            if r["split"] == sp:
                by[content_key(r)].append(r)
        for rs in by.values():
            if len(rs) < 2:
                continue
            if len({lab(r) for r in rs}) > 1:
                conflicts.append((sp, rs))
                continue
            rs = sorted(rs, key=lambda r: (prio.get(r.get("found_in"), 3), r["trajectory_id"]))
            for r in rs[1:]:
                drop.add(id(r))
                why[id(r)] = f"duplicate inside {sp}"
    # (d) D-026: identical rows inside the training pool (train + validation). Same label: keep one (a real-origin row first, then the smaller id). Conflicting labels
    # (the same input labeled correct and incorrect, or two classes): the input is pure label noise, so every row of the group is dropped and the group is listed.
    pool = defaultdict(list)
    for r in final:
        if r["split"] in ("train", "val") and id(r) not in drop:
            pool[content_key(r)].append(r)
    train_conflicts = []
    for rs in pool.values():
        if len(rs) < 2:
            continue
        if len({lab(r) for r in rs}) > 1:
            train_conflicts.append(rs)
            for r in rs:
                drop.add(id(r))
                why[id(r)] = "training input labeled in conflicting ways (all rows of the group dropped)"
            continue
        rs = sorted(rs, key=lambda r: (not r.get("real_origin"), r["split"] != "train", r["trajectory_id"]))
        for r in rs[1:]:
            drop.add(id(r))
            why[id(r)] = "duplicate inside the training pool"
    rem = Counter((why[id(r)], r["split"], lab(r)) for r in final if id(r) in drop)
    print(f"\ncontent-identical rows removed: {len(drop)}")
    for (w, sp, k), n in sorted(rem.items()):
        print(f"   {n:3d}  {k:9s} ({sp}): {w}")
    print(f"content-identical groups with CONFLICTING labels (left as they are, for a human): {len(conflicts)}")
    for rs in train_conflicts:
        print("   training pool, conflicting labels, all dropped: " + " | ".join(f"{r['trajectory_id'][-52:]} ({lab(r)}, {r['split']})" for r in rs))
    for sp, rs in conflicts:
        print(f"   {sp}: " + " | ".join(f"{r['trajectory_id'][-52:]} ({lab(r)}, {r.get('found_in')})" for r in rs))
    return [r for r in final if id(r) not in drop], conflicts


def tag_class7(final):
    """Every Class 7 row gets class7_reason and class7_reason_source: human (a queue-5 decision), generator (the state-flip rows are ineligible_state by
    construction) or heuristic (data/class7_reasons.py, structure only; 'unlabeled' when it finds nothing)."""
    src = Counter()
    for r in final:
        if r["label"] == "incorrect" and int(r["error_class"]) == 7:
            r["error_class_name"] = CLASS_NAMES[7]
            if not r.get("class7_reason"):
                if r.get("source") == "synthetic_state_flip" or str(r.get("source", "")).startswith("synthetic_state"):
                    r["class7_reason"], r["class7_reason_source"] = "ineligible_state", "generator"
                else:
                    pre = suggest(r["turns"], r["candidate_tool_call"], r["domain"])
                    r["class7_reason"], r["class7_reason_source"] = pre["reason"] or "unlabeled", "heuristic"
            src[(r["split"], r["class7_reason_source"])] += 1
    print("\nClass 7 rows by split and reason source:", dict(sorted(src.items())))
    for sp in ("test", "val", "train"):
        c = Counter((r["class7_reason"], r["class7_reason_source"]) for r in final if r["split"] == sp and r["label"] == "incorrect" and int(r["error_class"]) == 7)
        if c:
            print(f"   {sp}: {dict(sorted(c.items()))}")


# ───────────────────────────── 5. late additions: v3a and v3b (D-028) ─────────────────────────────
# Rows added AFTER the validation split is fixed, to the training split only, from task groups that are in neither validation, the held-out synthetic test nor the test
# set: so validation.jsonl, test.jsonl and test_synth.jsonl are byte-identical to the v2 build, and no group straddles two splits.

def success_row(rec, runs):
    msgs = runs.get(rec["run_key"], True)
    turns, call = history(msgs, rec["call_index"])
    return {"trajectory_id": f"tau2_success::{rec['run_key']}::C{rec['call_index']}", "domain": rec["domain"], "task_id": str(rec["task_id"]), "source_model": rec["source_model"],
            "tool_schemas_text": schema_text(rec["domain"]), "turns": turns, "candidate_tool_call": call, "gold_action": None, "label": "correct", "label_tier": "tau2_success_gated",
            "split": "train", "error_class": None, "error_class_name": None, "source": "tau2_success", "origin": "natural", "real_origin": True,
            "run_key": rec["run_key"], "call_index": rec["call_index"]}


def tool_balance(final, runs, excluded_groups, cap_fraction=0.25):
    """v3a: for every tool whose training errors outnumber its correct rows, add correct calls of that tool from SUCCESSFUL (reward 1) simulations of train-split tasks,
    until correct rows equal errors or the successful runs have no more eligible calls, worst-imbalanced tools first, within cap_fraction of the training size.
    A call is used only if it passes the same rules as the other correct rows (screen_positive: not an identical repeat, no structural Class 7 signal, not a wrong line lookup),
    did not return a tool error, is not a free-text tool, and its content is not already in any split."""
    from mine_real_errors import FREE_TEXT_TOOLS, agent_calls, is_error, iter_sims
    train = [r for r in final if r["split"] == "train"]
    E, C = Counter(), Counter()
    for r in train:
        (E if r["label"] == "incorrect" else C)[r["candidate_tool_call"]["name"]] += 1
    deficit = {t: E[t] - C[t] for t in E if E[t] > C[t]}
    split = json.load(open(DER / "real_split.json"))["tasks"]
    train_tasks = {d: {str(t) for t in split[d]["train"]} for d in split}
    cand = defaultdict(lambda: defaultdict(list))    # tool -> run_key -> calls of successful runs
    n_sims = n_ok = 0
    for s in iter_sims():
        d = s["domain"]
        if d not in train_tasks or str(s["task_id"]) not in train_tasks[d] or f"{d}::{s['task_id']}" in excluded_groups:
            continue
        n_sims += 1
        if ((s["reward_info"] or {}).get("reward") or 0) < 1:
            continue
        n_ok += 1
        k = f"{s['file']}::{s['sim_id']}"
        err_ids = {m.get("id") for m in s["messages"] if is_error(m)}
        for c in agent_calls(s["messages"]):
            if c["name"] in deficit and c["name"] not in FREE_TEXT_TOOLS and c["id"] not in err_ids:
                cand[c["name"]][k].append({"run_key": k, "call_index": c["k"], "domain": d, "task_id": str(s["task_id"]), "source_model": s["source_model"], "name": c["name"]})
    print(f"\ntool balancing (v3a): {n_ok} successful of {n_sims} simulations of train-split tasks outside validation, the held-out test and the test set; "
          f"{len(deficit)} tools have more training errors than correct rows (deficit {sum(deficit.values())})")
    existing = {content_key(r) for r in final}
    cap = int(cap_fraction * len(train))
    order = sorted(deficit, key=lambda t: (-E[t] / (C[t] + 1), -deficit[t], t))      # worst-imbalanced first
    added, stats = [], {}
    for t in order:
        if len(added) >= cap:
            stats[t] = dict(want=deficit[t], avail=sum(len(v) for v in cand[t].values()), added=0, why="cap")
            continue
        rng = random.Random(f"{SEED}-balance-{t}")
        runs_t = sorted(cand[t])
        rng.shuffle(runs_t)
        pool, i = [], 0                                  # round-robin over runs, so that no run dominates
        while any(i < len(cand[t][k]) for k in runs_t):
            pool += [cand[t][k][i] for k in runs_t if i < len(cand[t][k])]
            i += 1
        avail = len(pool)
        runs.preload({r["run_key"] for r in pool[: 4 * deficit[t] + 20]})
        got, why = 0, Counter()
        for rec in pool:
            if got >= deficit[t] or len(added) >= cap:
                break
            reason = screen_positive(rec, "natural", runs)
            if reason:
                why["rules: " + reason.split(" (")[0]] += 1
                continue
            row = success_row(rec, runs)
            ck = content_key(row)
            if ck in existing:
                why["content already in the dataset"] += 1
                continue
            existing.add(ck)
            added.append(row)
            got += 1
        stats[t] = dict(want=deficit[t], avail=avail, added=got, why=dict(why))
    print(f"   {'tool':34s} {'errors':>6s} {'correct':>7s} {'deficit':>7s} {'available':>9s} {'added':>5s} {'correct after':>13s}  excluded")
    for t in order:
        s = stats[t]
        if s["avail"] or s["added"]:
            print(f"   {t:34s} {E[t]:6d} {C[t]:7d} {deficit[t]:7d} {s['avail']:9d} {s['added']:5d} {C[t] + s['added']:13d}  {s['why'] if s['why'] else ''}")
    none = [t for t in order if not stats[t]["avail"]]
    print(f"   tools with a deficit and no successful-run calls (BFCL, CriticTool and other non-tau2 tools): {len(none)} ({sum(deficit[t] for t in none)} rows of deficit)")
    print(f"   rows added: {len(added)} = {len(added) / len(train):.1%} of the {len(train)} training rows (cap {cap_fraction:.0%} = {cap})")
    return added


def late_candidate_rows(items, final, runs, excluded_groups, screen_errors=True):
    """v3b: the first-labeler candidates of the newly localized train runs (batch train_more), through the D-026 gate, with their earlier calls as correct rows."""
    out, stats = [], Counter()
    pool = items["natural"]
    existing = {content_key(r) for r in final}
    keep_cands, keep_ids = {}, set()
    for c in pool["cands"]:
        if c.get("batch") != "train_more" or c["split"] != "train":
            continue
        k = ("natural", int(c["proposed_class"]))
        if k not in TRAIN_FIRST_LABELER:
            stats[f"candidate held back: class {int(c['proposed_class'])} is not admitted"] += 1
            continue
        if f"{c['domain']}::{c['task_id']}" in excluded_groups:
            stats["candidate left out: its task group is in validation, the held-out test or the test set"] += 1
            continue
        keep_ids.add(c["candidate_id"])
        why = screen_candidate(c, "natural", runs) if screen_errors else None
        if why:
            stats[f"error not used (class {int(c['proposed_class'])}): {why}"] += 1
            continue
        row = review_row("natural", "error", c, "incorrect", int(c["proposed_class"]), "train", runs, tier="first_labeler_gated")
        row["real_origin"], row["batch"] = True, "train_more"
        ck = content_key(row)
        if ck in existing:
            stats["error left out: content already in the dataset"] += 1
            continue
        existing.add(ck)
        out.append(row)
        stats[f"error admitted (class {int(c['proposed_class'])}, {c['domain']})"] += 1
    for p in pool["pos"]:
        if p.get("batch") != "train_more" or p["split"] != "train" or p.get("linked_candidate_id") not in keep_ids:
            continue
        why = screen_positive(p, "natural", runs)
        if why:
            stats["correct row excluded by the rules: " + why.split(" (")[0]] += 1
            continue
        row = review_row("natural", "correct", p, "correct", None, "train", runs, tier="linked_positive_gated")
        row["real_origin"], row["batch"] = True, "train_more"
        ck = content_key(row)
        if ck in existing:
            stats["correct row left out: content already in the dataset"] += 1
            continue
        existing.add(ck)
        out.append(row)
        stats["correct row admitted"] += 1
    print("\nnew real rows from the newly localized train runs (v3b):")
    for k, n in sorted(stats.items()):
        print(f"   {n:5d}  {k}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default=str(DER / "review_queue_decisions.jsonl"))
    ap.add_argument("--train-positives", choices=["linked", "accepted", "gated"], default="linked", help="linked (default, D-026): the pending positives of each admitted train error; accepted: only human-accepted ones (the D-022 behaviour); gated: by the human-confirmed share")
    ap.add_argument("--no-error-rules", action="store_true", help="do not apply the read-only / line-probe rules to the first-labeler train errors (the literal D-026 step 1; default: apply them)")
    ap.add_argument("--variant", choices=sorted(VARIANT_DIRS), default="v2", help="v2 (default): the D-026 dataset; v3a: + tool balancing; v3b: v3a + the newly localized train runs (D-028)")
    ap.add_argument("--preview", action="store_true", help="write the current dry-run result to data/derived/final_preview/ (gitignored, NOT the final dataset)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--review-complete", action="store_true")
    a = ap.parse_args()
    if a.write and not a.review_complete:
        sys.exit("refusing to write final files: pass --review-complete only once the human review is finished")

    dec = decisions(Path(a.decisions))
    print(f"decisions file {a.decisions}: {len(dec)} decisions")
    items = review_pool()
    runs = Runs()

    rows = base_rows()
    rows, dropped = drop_new_test_tasks(rows)
    print(f"\ncurrent 7,580 training rows minus new test-split tasks (D-018): dropped {sum(dropped.values())} rows -> {len(rows)} kept")
    for (dom, src), n in sorted(dropped.items()):
        print(f"   dropped {n:4d}  {dom:8s} {src}")

    rows, held, lost, groups = hold_out_synthetic(rows)
    print(f"\nheld-out synthetic test: {len(groups)} task groups; {len(held)} rows become test_synth "
          f"(negatives {dict(sorted(Counter(r['error_class'] for r in held if r['label'] == 'incorrect').items()))}, "
          f"paired controls {sum(r['label'] == 'correct' for r in held)}); {len(lost)} other rows of those groups are dropped to avoid leakage")

    gate = class_gate(items, dec)
    pgate = positive_gate(items, dec)
    print("\ntrain gate for review-pool labels: TRAIN_AUTO_ACCEPT (natural classes 1, 4, 7; induced class 5); everything else needs a human review (queue 3)")
    print("   queue-1/4 human decisions on test candidates, per (origin, class): the policy is fixed, the data rule (>= 8 reviewed, >= 70% confirmed) is shown for comparison")
    for (o, c), (n, ok, admitted) in sorted(gate.items()):
        data_rule = n >= GATE_MIN_REVIEWED and ok / max(n, 1) >= GATE_PRECISION
        print(f"   {o:8s} class {c} {CLASS_NAMES[c]:22s}: {ok:2d}/{n:2d} confirmed ({(ok / n if n else 0):4.0%}) -> train labels {'AUTO-ACCEPTED' if admitted else 'dropped (unless reviewed)'}"
              + ("" if data_rule == admitted else "   <-- the data rule would decide differently"))
    print("positive gate (queue-2 decisions, only used with --train-positives gated):", {o: f"{ok}/{n}" for o, (n, ok, _) in pgate.items()})

    q5 = [i for i in load(DER / "review_queue.jsonl") if i["queue"] == 5]
    q5_pending = {i["orig_item_id"] for i in q5} - {k for k, d in dec.items() if "q5" in d.get("superseded_chain", [])}
    if q5:
        print(f"\nqueue 5 (Class 7 / read-only re-review): {len(q5)} items, {len(q5) - len(q5_pending)} decided and applied over the earlier decision, {len(q5_pending)} waiting")
    runs.preload({c["run_key"] for c in items["natural"]["cands"]} | {p["run_key"] for p in items["natural"]["pos"] if p["split"] == "train"})
    r_test, r_train, notes = reviewed_rows(items, dec, gate, pgate, a.train_positives, runs, q5_pending, screen_errors=not a.no_error_rules)
    print("\nreview pool:")
    for k, n in sorted(notes.items()):
        print(f"   {n:5d}  {k}")
    # induced errors and paired handlings enter together or not at all
    ind_err = [r for r in r_train if r["origin"] == "induced" and r["label"] == "incorrect"]
    ind_pos = [r for r in r_train if r["origin"] == "induced" and r["label"] == "correct"]
    if ind_err and not ind_pos:
        print(f"   pair rule: {len(ind_err)} induced train errors held back because no paired correct handling is admitted")
        r_train = [r for r in r_train if r not in ind_err]

    flips = tagged(load(DER / "class7_stateflip.jsonl"), None, None, "train")
    in_held = [r for r in flips if group_of(r) in set(groups)]
    flips = [r for r in flips if group_of(r) not in set(groups)]
    if in_held:
        print(f"\nstate flips: {len(in_held)} dropped because their task group was moved to the held-out synthetic test")
    extra = tagged(load(DER / "external_critictool.jsonl"), "critictool", "external", "train") + flips
    test_real = tagged(load(DER / "test_real.jsonl"), "human_labeled_phase4", "natural", "test")
    for r in test_real:
        r["found_in"], r["run_key"] = "test_real", r["trajectory_id"]
    test_real = apply_queue5_to_test_real(test_real, dec)
    test_real, r_test = dedupe_test(test_real, r_test)

    held_groups = {group_of(r) for r in held}
    moved = [r for r in r_train if group_of(r) in held_groups]
    if moved:  # a review-pool train row in a task group that became held-out synthetic test would straddle train and test_synth
        print(f"\nreview-pool train rows dropped because their task group was moved to the held-out synthetic test: {len(moved)} {dict(Counter((r['origin'], r['label'], r.get('error_class')) for r in moved))}")
        r_train = [r for r in r_train if group_of(r) not in held_groups]
    train_all = [dict_arguments(r) for r in rows + r_train + extra]
    wrapped = sum(bool(r.get("arguments_wrapped")) for r in train_all)
    if wrapped:
        print(f"\nnon-dict candidate arguments wrapped as {{\"text\": ...}}: {wrapped} rows (CriticTool free-text answers)")
    for r in train_all:
        r.pop("_file", None)
    train_all = assign_val(train_all)
    final = train_all + held + [dict_arguments(r) for r in test_real + r_test]
    if a.variant in ("v3a", "v3b"):
        excluded = {group_of(r) for r in final if r["split"] != "train"} | held_groups
        late = tool_balance(final, runs, excluded)
        if a.variant == "v3b":
            late += late_candidate_rows(items, final + late, runs, excluded, screen_errors=not a.no_error_rules)
        final = final + [dict_arguments(r) for r in late]
    final, conflicts = dedupe_content(final)
    tag_class7(final)

    # checks
    assert all(r.get("source") and r.get("origin") for r in final), "every row needs source and origin"
    assert not [r for r in final if r["split"] in ("test",) and (r["origin"] == "synthetic" or str(r["source"]).startswith("synthetic"))], "a synthetic row is in test"
    assert all(r["origin"] == "synthetic" for r in held)
    ids = [r["trajectory_id"] for r in final]
    assert len(ids) == len(set(ids)), "duplicate trajectory ids"
    g_split = defaultdict(set)
    for r in final:
        if r["domain"] != "critictool":
            g_split[group_of(r)].add("train" if r["split"] in ("train", "val") else r["split"])
    straddle = [g for g, s in g_split.items() if len(s) > 1]
    assert not straddle, f"task groups in more than one of train/test/test_synth: {straddle[:5]}"
    negs = {r["trajectory_id"]: r["split"] for r in final if r["label"] == "incorrect"}
    split_of = {r["trajectory_id"]: r["split"] for r in final}
    assert all(split_of.get(r["paired_negative_id"]) == r["split"] for r in final if r.get("paired_negative_id")), "a control is separated from its negative"
    print(f"\nchecks passed: source/origin on every row, no synthetic test row, unique ids, no task group in two of train/test/test_synth, controls stay with their negatives")

    table(final, lambda r: f"class {class_of(r)}" if r["label"] == "incorrect" else "correct", "rows per class and split")
    table(final, lambda r: r["source"], "rows per source and split")
    table(final, lambda r: r["origin"], "rows per origin and split")
    table(final, lambda r: r["domain"], "rows per domain and split")
    inc = Counter(r["split"] for r in final if r["label"] == "incorrect")
    print("\nincorrect share:", {s: f"{inc[s]}/{n}" for s, n in sorted(Counter(r['split'] for r in final).items())})

    # what the review pool could add once labels exist (for planning only)
    pend = [(o, c) for o, p in items.items() for c in p["cands"] if c["split"] == "train" and c.get("auto_status") == "auto_accepted"]
    print("\nplanning: train-split auto-accepted candidates waiting for the gate:", dict(Counter((o, int(c['proposed_class'])) for o, c in pend)))
    print("planning: test-split candidates and correct examples waiting for review:",
          {o: (sum(c['split'] == 'test' for c in p['cands']), sum(x['split'] == 'test' for x in p['pos'])) for o, p in items.items()}, "(errors, correct)")

    if a.write or a.preview:
        out = (DER / VARIANT_DIRS[a.variant]) if a.write else DER / "final_preview"
        out.mkdir(parents=True, exist_ok=True)
        for s in ("train", "val", "test", "test_synth"):
            with open(out / f"{s}.jsonl", "w") as fh:
                for r in final:
                    if r["split"] == s:
                        fh.write(json.dumps(r) + "\n")
        if a.preview:
            (out / "PREVIEW.txt").write_text(f"Dry-run output of build_final_dataset.py with {len(dec)} review decisions. NOT the final dataset: the review is not complete.\n")
        print("wrote", out)
    else:
        print("\ndry run: nothing written")


if __name__ == "__main__":
    main()
