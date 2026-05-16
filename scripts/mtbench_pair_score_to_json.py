"""Convert FastChat MT-Bench *pairwise* judgment JSONL into per-model + scoreboard JSON.

For pairwise-baseline mode, each (question, turn) is judged twice with positions
swapped (g1_winner, g2_winner) to neutralize position bias. Following FastChat's
show_result.py:
  - If g1_winner == "tie" OR g1_winner != g2_winner -> tie
  - Else (both judges agree on a non-tie winner) -> that model wins, other loses
  - "error" judgments are dropped

Per-model summary: win_rate = (wins + 0.5*ties) / (wins + losses + ties)
"""
import argparse
import datetime
import json
import os
from collections import defaultdict
from pathlib import Path


def family_of(repo_id):
    rid = repo_id.lower()
    if "qwen3-1.7b" in rid: return "qwen3-1.7b"
    if "qwen3-8b" in rid: return "qwen3-8b"
    if "llama-3.2-1b" in rid or "llama3.2-1b" in rid: return "llama-3.2-1b"
    if "llama-3.1-8b" in rid: return "llama-3.1-8b"
    return "other"


def safe_filename(repo_id):
    return repo_id.replace("/", "__")


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_question_category(question_file):
    cat = {}
    with open(question_file) as f:
        for line in f:
            q = json.loads(line)
            cat[q["question_id"]] = q["category"]
    return cat


def load_pair_judgments(jsonl_path, model_id, baseline_id, judge_model):
    """Return list of (qid, turn, outcome, cat_aware) where outcome in {win, loss, tie}.

    Dedupes by (qid, turn) keeping last record. Matches either ordering
    (model_id can be model_1 or model_2).
    """
    if not os.path.exists(jsonl_path):
        return []
    seen = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            judge = rec.get("judge")
            judge_name = judge[0] if isinstance(judge, list) and judge else judge
            if judge_name != judge_model:
                continue
            m1, m2 = rec.get("model_1"), rec.get("model_2")
            if model_id not in (m1, m2) or baseline_id not in (m1, m2):
                continue
            if m1 == m2:  # shouldn't happen
                continue
            key = (rec.get("question_id"), rec.get("turn"), tuple(sorted([m1, m2])))
            seen[key] = rec
    out = []
    for (qid, turn, _), rec in seen.items():
        g1 = rec.get("g1_winner")
        g2 = rec.get("g2_winner")
        if g1 == "error" or g2 == "error":
            continue
        m1, m2 = rec["model_1"], rec["model_2"]
        if g1 == "tie" or g1 != g2:
            outcome = "tie"
        else:
            winner = m1 if g1 == "model_1" else m2
            outcome = "win" if winner == model_id else "loss"
        out.append((qid, turn, outcome))
    return out


def compute_summary(judgments, cat_map):
    if not judgments:
        return None
    win = loss = tie = 0
    by_turn = defaultdict(lambda: [0, 0, 0])  # win, loss, tie
    by_cat = defaultdict(lambda: [0, 0, 0])
    for qid, turn, outcome in judgments:
        if outcome == "win":
            win += 1; by_turn[turn][0] += 1
        elif outcome == "loss":
            loss += 1; by_turn[turn][1] += 1
        else:
            tie += 1; by_turn[turn][2] += 1
        cat = cat_map.get(qid, "unknown")
        idx = {"win": 0, "loss": 1, "tie": 2}[outcome]
        by_cat[cat][idx] += 1
    total = win + loss + tie
    def wr(w, l, t):
        tot = w + l + t
        return round((w + 0.5 * t) / tot, 4) if tot else None
    return {
        "win_rate": wr(win, loss, tie),
        "win_rate_turn1": wr(*by_turn.get(1, [0, 0, 0])),
        "win_rate_turn2": wr(*by_turn.get(2, [0, 0, 0])),
        "wins": win,
        "losses": loss,
        "ties": tie,
        "n_total": total,
        "by_category": {c: {"wr": wr(*v), "wins": v[0], "losses": v[1], "ties": v[2]} for c, v in sorted(by_cat.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--baseline-model", default="gpt-3.5-turbo")
    ap.add_argument("--judgment-file", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--judge-model", default="gpt-4")
    ap.add_argument("--out-dir", default="results/eval/mtbench-pair")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    cat_map = load_question_category(args.question_file)
    judgments = load_pair_judgments(args.judgment_file, args.model_id, args.baseline_model, args.judge_model)
    summary = compute_summary(judgments, cat_map)
    if summary is None:
        print(f"[mtbench-pair-score] no judgments for {args.model_id} vs {args.baseline_model}; skipping")
        return 0

    repo_id = args.repo_id or args.model_id
    family = family_of(repo_id)
    summary["family"] = family
    summary["repo_id"] = repo_id
    summary["model_id"] = args.model_id
    summary["baseline_model"] = args.baseline_model
    summary["judge_model"] = args.judge_model
    summary["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    if args.label:
        summary["label"] = args.label

    out_dir = Path(args.out_dir)
    per_file = out_dir / family / "per_model" / f"{safe_filename(repo_id)}.json"
    atomic_write_json(per_file, summary)
    print(f"[mtbench-pair-score] wrote {per_file} (win_rate={summary['win_rate']}, "
          f"W/L/T={summary['wins']}/{summary['losses']}/{summary['ties']})")

    score_file = out_dir / "scores.json"
    if score_file.exists():
        with open(score_file) as f:
            board = json.load(f)
    else:
        board = {
            "bench": "mt_bench",
            "mode": "pairwise-baseline",
            "baseline_model": args.baseline_model,
            "judge_model": args.judge_model,
            "started_at": datetime.datetime.utcnow().isoformat() + "Z",
            "models": {},
        }
    board["last_update"] = datetime.datetime.utcnow().isoformat() + "Z"
    board["models"][repo_id] = summary
    atomic_write_json(score_file, board)
    print(f"[mtbench-pair-score] updated {score_file} (n_models={len(board['models'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
