"""Convert AlpacaEval annotations into per-model JSON + global scoreboard JSON.

alpaca_eval CLI writes:
  - annotations: <output_dir>/<generator>/annotations.json  (per-instance scores)
  - leaderboard: <output_dir>/<generator>/leaderboard.csv   (per-model aggregate)

This script reads the leaderboard CSV for the given --model-id, extracts win_rate,
lc_win_rate (length-controlled), and standard_error fields, then writes:
  - results/eval/alpaca/<family>/per_model/<safe_id>.json
  - results/eval/alpaca/scores.json
"""
import argparse
import csv
import datetime
import json
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="generator field used in model_outputs JSON")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--leaderboard-csv", required=True,
                    help="path to <output_dir>/<generator>/leaderboard.csv")
    ap.add_argument("--annotator", default="weighted_alpaca_eval_gpt4_turbo",
                    help="LC v2 annotator name; written into output JSON")
    ap.add_argument("--out-dir", default="results/eval/alpaca")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.leaderboard_csv):
        print(f"[alpaca-score] no leaderboard CSV at {args.leaderboard_csv}; skipping")
        return 0
    with open(args.leaderboard_csv) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows:
        print(f"[alpaca-score] empty leaderboard CSV; skipping")
        return 0

    # The leaderboard may list multiple models if accumulated; find ours.
    row = None
    for r in rows:
        if r.get("") == args.model_id or r.get("model") == args.model_id:
            row = r
            break
    if row is None:
        # Fallback: take the first row (single-model case)
        if len(rows) == 1:
            row = rows[0]
        else:
            print(f"[alpaca-score] could not find model_id={args.model_id} in CSV")
            return 1

    def to_float(s):
        try:
            return round(float(s), 4)
        except (ValueError, TypeError):
            return None

    repo_id = args.repo_id or args.model_id
    family = family_of(repo_id)
    summary = {
        "win_rate": to_float(row.get("win_rate")),
        "lc_win_rate": to_float(row.get("length_controlled_winrate")),
        "standard_error": to_float(row.get("standard_error")),
        "avg_length": to_float(row.get("avg_length")),
        "n_total": int(float(row.get("n_total", 0) or 0)) if row.get("n_total") else None,
        "n_wins": int(float(row.get("n_wins", 0) or 0)) if row.get("n_wins") else None,
        "n_draws": int(float(row.get("n_draws", 0) or 0)) if row.get("n_draws") else None,
        "n_losses": int(float(row.get("n_losses", 0) or 0)) if row.get("n_losses") else None,
        "family": family,
        "repo_id": repo_id,
        "model_id": args.model_id,
        "annotator": args.annotator,
        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if args.label:
        summary["label"] = args.label

    out_dir = Path(args.out_dir)
    per_file = out_dir / family / "per_model" / f"{safe_filename(repo_id)}.json"
    atomic_write_json(per_file, summary)
    print(f"[alpaca-score] wrote {per_file} (win_rate={summary['win_rate']}, "
          f"lc_win_rate={summary['lc_win_rate']})")

    score_file = out_dir / "scores.json"
    if score_file.exists():
        with open(score_file) as f:
            board = json.load(f)
    else:
        board = {
            "bench": "alpaca_eval",
            "annotator": args.annotator,
            "n_prompts": 805,
            "started_at": datetime.datetime.utcnow().isoformat() + "Z",
            "models": {},
        }
    board["last_update"] = datetime.datetime.utcnow().isoformat() + "Z"
    board["models"][repo_id] = summary
    atomic_write_json(score_file, board)
    print(f"[alpaca-score] updated {score_file} (n_models={len(board['models'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
