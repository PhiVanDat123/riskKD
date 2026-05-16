"""Convert FastChat MT-Bench judgment JSONL into per-model JSON + global scoreboard JSON.

After gen_judgment writes judgments to model_judgment/gpt-4_single.jsonl (one record per
question-turn-model), this script:
 1. Reads judgments for the given --model-id
 2. Loads question categories from question.jsonl
 3. Computes overall mean, turn-1 mean, turn-2 mean, per-category mean
 4. Writes results/eval/mtbench/<family>/per_model/<safe_id>.json (atomic)
 5. Merges into results/eval/mtbench/scores.json (atomic)
"""
import argparse
import datetime
import json
import os
from collections import defaultdict
from pathlib import Path


def load_judgments(jsonl_path, model_id, judge_model):
    """Read judgments and dedupe by (question_id, turn), keeping the most-recent record.
    The same (qid, turn) can appear multiple times if gen_judgment is re-run (e.g. after
    a smoke test that populated the JSONL with a subset of questions).
    """
    seen = {}
    if not os.path.exists(jsonl_path):
        return []
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("model") != model_id:
                continue
            judge = rec.get("judge")
            judge_name = judge[0] if isinstance(judge, list) and judge else judge
            if judge_name != judge_model:
                continue
            key = (rec.get("question_id"), rec.get("turn"))
            seen[key] = rec  # last-write-wins
    return list(seen.values())


def load_question_category(question_file):
    cat = {}
    with open(question_file) as f:
        for line in f:
            q = json.loads(line)
            cat[q["question_id"]] = q["category"]
    return cat


def family_of(repo_id):
    rid = repo_id.lower()
    if "qwen3-1.7b" in rid: return "qwen3-1.7b"
    if "qwen3-8b" in rid: return "qwen3-8b"
    if "llama-3.2-1b" in rid or "llama3.2-1b" in rid: return "llama-3.2-1b"
    if "llama-3.1-8b" in rid: return "llama-3.1-8b"
    return "other"


def safe_filename(repo_id):
    return repo_id.replace("/", "__")


def compute_summary(judgments, cat_map):
    if not judgments:
        return None
    scores = []
    by_turn = defaultdict(list)
    by_cat = defaultdict(list)
    for j in judgments:
        s = j.get("score")
        if s is None or s < 0:
            continue
        qid = j.get("question_id")
        t = j.get("turn", 1)
        scores.append(s)
        by_turn[t].append(s)
        cat = cat_map.get(qid, "unknown")
        by_cat[cat].append(s)
    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None
    return {
        "overall": avg(scores),
        "turn1": avg(by_turn.get(1, [])),
        "turn2": avg(by_turn.get(2, [])),
        "by_category": {c: avg(v) for c, v in sorted(by_cat.items())},
        "n_judged": len(scores),
    }


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="model_id used in the answer JSONL")
    ap.add_argument("--repo-id", default=None, help="HF repo id; defaults to model_id")
    ap.add_argument("--judgment-file", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--judge-model", default="gpt-4")
    ap.add_argument("--out-dir", default="results/eval/mtbench")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    cat_map = load_question_category(args.question_file)
    judgments = load_judgments(args.judgment_file, args.model_id, args.judge_model)
    summary = compute_summary(judgments, cat_map)
    if summary is None:
        print(f"[mtbench-score] no judgments for {args.model_id}; skipping")
        return 0

    repo_id = args.repo_id or args.model_id
    family = family_of(repo_id)
    summary["family"] = family
    summary["repo_id"] = repo_id
    summary["model_id"] = args.model_id
    summary["judge_model"] = args.judge_model
    summary["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    if args.label:
        summary["label"] = args.label

    out_dir = Path(args.out_dir)
    per_file = out_dir / family / "per_model" / f"{safe_filename(repo_id)}.json"
    atomic_write_json(per_file, summary)
    print(f"[mtbench-score] wrote {per_file} (overall={summary['overall']}, n_judged={summary['n_judged']})")

    score_file = out_dir / "scores.json"
    if score_file.exists():
        with open(score_file) as f:
            board = json.load(f)
    else:
        board = {
            "bench": "mt_bench",
            "judge_model": args.judge_model,
            "judge_mode": "single",
            "n_questions": 80,
            "started_at": datetime.datetime.utcnow().isoformat() + "Z",
            "models": {},
        }
    board["last_update"] = datetime.datetime.utcnow().isoformat() + "Z"
    board["models"][repo_id] = summary
    atomic_write_json(score_file, board)
    print(f"[mtbench-score] updated {score_file} (n_models={len(board['models'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
