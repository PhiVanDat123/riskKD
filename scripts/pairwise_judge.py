"""Pairwise MT-Bench judging with Sonnet 4.6 + gpt-5.4-mini.

Reuses existing MT-Bench answer JSONLs (no model generation). For each (anchor,
baseline) pair, for each (question, turn), runs TWO calls per judge with positions
swapped to neutralize position bias (FastChat convention):
  - g1: A=anchor's answer, B=baseline's answer
  - g2: A=baseline's answer, B=anchor's answer

Win/tie/loss per pair = matched-position-swap rule:
  - anchor wins   <=> g1 == 'A' AND g2 == 'B'  (both judges agree anchor better)
  - baseline wins <=> g1 == 'B' AND g2 == 'A'  (both judges agree baseline better)
  - tie           <=> any other combination (incl. 'C' votes or disagreement)

Per FastChat: math/reasoning/coding categories use a pair-math-v1 prompt with
reference answers (but we use the simpler pair-v2 prompt uniformly; pair-math
adds a reference-answer field but the judge call structure is the same).
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from pathlib import Path


# FastChat pair-v2 prompt (extracted from /workspace/FastChat/.../data/judge_prompts.jsonl)
PAIR_V2_SYSTEM = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by "
    "two AI assistants to the user question displayed below. You should choose the assistant "
    "that follows the user's instructions and answers the user's question better. Your "
    "evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, "
    "creativity, and level of detail of their responses. Begin your evaluation by comparing "
    "the two responses and provide a short explanation. Avoid any position biases and ensure "
    "that the order in which the responses were presented does not influence your decision. "
    "Do not allow the length of the responses to influence your evaluation. Do not favor "
    "certain names of the assistants. Be as objective as possible. After providing your "
    "explanation, output your final verdict by strictly following this format: \"[[A]]\" if "
    "assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."
)
PAIR_V2_USER_TPL = (
    "[User Question]\n{question}\n\n"
    "[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
)

VERDICT_RE = re.compile(r"\[\[([ABC])\]\]")


def parse_verdict(text):
    """Return 'A', 'B', 'C' (tie), or None (couldn't parse)."""
    if not text:
        return None
    m = VERDICT_RE.findall(text)
    return m[-1] if m else None


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_answers(answer_dir, model_id):
    """Return dict {question_id: [turn1_answer, turn2_answer]}."""
    path = Path(answer_dir) / f"{model_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"answers not found: {path}")
    out = {}
    for rec in load_jsonl(path):
        turns = rec["choices"][0]["turns"]
        out[rec["question_id"]] = turns
    return out


def build_user_prompt(question, answer_a, answer_b):
    return PAIR_V2_USER_TPL.format(question=question, answer_a=answer_a, answer_b=answer_b)


# --- judge dispatchers -------------------------------------------------------

def judge_anthropic(model, messages_user, system, max_tokens=512):
    import anthropic
    client = anthropic.Anthropic()
    r = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": messages_user}],
    )
    return r.content[0].text, {
        "input_tokens": r.usage.input_tokens,
        "output_tokens": r.usage.output_tokens,
    }


def judge_openai(model, messages_user, system, max_tokens=512):
    from openai import OpenAI
    client = OpenAI()
    # gpt-5.x and reasoning-family models require max_completion_tokens and don't
    # accept the older max_tokens. Probe by model name prefix.
    is_reasoning = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4")
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": messages_user},
        ],
    }
    if is_reasoning:
        # Bigger budget for reasoning models even if they sometimes don't use it
        kwargs["max_completion_tokens"] = max(max_tokens, 2048)
    else:
        kwargs["max_tokens"] = max_tokens
    r = client.chat.completions.create(**kwargs)
    return r.choices[0].message.content or "", {
        "input_tokens": r.usage.prompt_tokens,
        "output_tokens": r.usage.completion_tokens,
    }


def call_judge(judge_model, system, user, max_tokens=512, max_retries=3):
    """Dispatch by model name. Returns (text, usage_dict) or raises."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            if judge_model.startswith("claude-"):
                return judge_anthropic(judge_model, user, system, max_tokens)
            else:
                return judge_openai(judge_model, user, system, max_tokens)
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise last_exc


# --- main pairwise loop ------------------------------------------------------

def judge_one_pair(args_tuple):
    """One (question, turn, position) call. Returns dict with verdict + usage."""
    (qid, turn, position, question_text, anchor_text, baseline_text, judge_model) = args_tuple
    if position == "g1":   # anchor is A, baseline is B
        a, b = anchor_text, baseline_text
    else:                   # swapped: baseline is A, anchor is B
        a, b = baseline_text, anchor_text
    user = build_user_prompt(question_text, a, b)
    try:
        text, usage = call_judge(judge_model, PAIR_V2_SYSTEM, user)
        verdict = parse_verdict(text)
    except Exception as e:
        return {"question_id": qid, "turn": turn, "position": position,
                "verdict": None, "error": str(e), "usage": None}
    return {
        "question_id": qid, "turn": turn, "position": position,
        "verdict": verdict, "judgment": text, "usage": usage,
    }


def aggregate(judgments):
    """From per-call records, group by (qid, turn) and compute outcome.

    FastChat rule: outcome = anchor_win iff g1==A AND g2==B; baseline_win iff g1==B AND g2==A;
    else tie (including parse errors or 'C' votes).
    """
    by_pair = {}
    for j in judgments:
        key = (j["question_id"], j["turn"])
        by_pair.setdefault(key, {})[j["position"]] = j["verdict"]
    wins, losses, ties, errors = 0, 0, 0, 0
    by_turn = {1: [0, 0, 0], 2: [0, 0, 0]}  # w, l, t per turn
    outcomes = []
    for (qid, turn), v in by_pair.items():
        g1 = v.get("g1")
        g2 = v.get("g2")
        if g1 is None or g2 is None:
            errors += 1; continue
        if g1 == "A" and g2 == "B":
            outcome = "anchor_win"; wins += 1; by_turn[turn][0] += 1
        elif g1 == "B" and g2 == "A":
            outcome = "baseline_win"; losses += 1; by_turn[turn][1] += 1
        else:
            outcome = "tie"; ties += 1; by_turn[turn][2] += 1
        outcomes.append({"question_id": qid, "turn": turn, "g1": g1, "g2": g2, "outcome": outcome})
    total = wins + losses + ties
    win_rate = (wins + 0.5 * ties) / total if total else None
    return {
        "wins": wins, "losses": losses, "ties": ties, "errors": errors,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "n_total": total,
        "win_rate_turn1": round((by_turn[1][0] + 0.5 * by_turn[1][2]) / max(sum(by_turn[1]), 1), 4),
        "win_rate_turn2": round((by_turn[2][0] + 0.5 * by_turn[2][2]) / max(sum(by_turn[2]), 1), 4),
        "outcomes": outcomes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True, help="anchor model_id (matches answer JSONL filename)")
    ap.add_argument("--baseline", required=True, help="baseline model_id")
    ap.add_argument("--judge", required=True, help="e.g. 'claude-sonnet-4-6' or 'gpt-5.4-mini'")
    ap.add_argument("--answer-dir", required=True, help="dir with <model_id>.jsonl files")
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--output", required=True, help="JSON output path")
    ap.add_argument("--num-questions", type=int, default=None, help="for smoke")
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--resume", action="store_true", help="skip if output already has the full aggregate")
    args = ap.parse_args()

    if args.resume and Path(args.output).exists():
        try:
            existing = json.load(open(args.output))
            if existing.get("aggregate", {}).get("n_total"):
                print(f"[pairwise] SKIP (resume) {args.output} already done")
                return 0
        except Exception:
            pass

    questions = load_jsonl(args.question_file)
    if args.num_questions:
        questions = questions[: args.num_questions]
    anchor_ans = load_answers(args.answer_dir, args.anchor)
    base_ans = load_answers(args.answer_dir, args.baseline)

    tasks = []
    for q in questions:
        qid = q["question_id"]
        if qid not in anchor_ans or qid not in base_ans:
            continue
        for turn in (1, 2):
            anchor_turn = anchor_ans[qid][turn - 1]
            base_turn = base_ans[qid][turn - 1]
            # FastChat's multi-turn pair judge actually wraps both turns into one user prompt;
            # for simplicity we judge each turn separately (gives 2x judgments per question,
            # which is what we want for win-rate-by-turn). Question text for turn 2 uses
            # both turn questions concatenated, mirroring FastChat single-grade convention.
            if turn == 1:
                question_text = q["turns"][0]
                a_text = anchor_turn
                b_text = base_turn
            else:
                question_text = (
                    f"### Turn 1:\n{q['turns'][0]}\n\n### Turn 2 (follow-up):\n{q['turns'][1]}"
                )
                a_text = f"### Turn 1:\n{anchor_ans[qid][0]}\n\n### Turn 2:\n{anchor_turn}"
                b_text = f"### Turn 1:\n{base_ans[qid][0]}\n\n### Turn 2:\n{base_turn}"
            for pos in ("g1", "g2"):
                tasks.append((qid, turn, pos, question_text, a_text, b_text, args.judge))

    print(f"[pairwise] {args.anchor} vs {args.baseline} | judge={args.judge} | "
          f"{len(tasks)} calls ({len(questions)} questions x 2 turns x 2 swaps)",
          flush=True)

    t0 = time.time()
    raw = []
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for i, result in enumerate(ex.map(judge_one_pair, tasks), 1):
            raw.append(result)
            if i % 50 == 0:
                print(f"  [{i}/{len(tasks)}] {time.time()-t0:.0f}s elapsed", flush=True)
    print(f"[pairwise] done in {time.time()-t0:.1f}s", flush=True)

    agg = aggregate(raw)
    total_in = sum((r.get("usage") or {}).get("input_tokens", 0) for r in raw)
    total_out = sum((r.get("usage") or {}).get("output_tokens", 0) for r in raw)
    out = {
        "anchor": args.anchor,
        "baseline": args.baseline,
        "judge": args.judge,
        "n_questions": len(questions),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "aggregate": agg,
        "raw": raw,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.output).with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.output)
    print(f"[pairwise] win={agg['wins']} loss={agg['losses']} tie={agg['ties']} err={agg['errors']} "
          f"win_rate={agg['win_rate']} -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
