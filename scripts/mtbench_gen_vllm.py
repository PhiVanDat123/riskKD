"""MT-Bench answer generation using vLLM offline batching.

Produces FastChat-compatible JSONL: {question_id, answer_id, model_id, choices, tstamp}.

Two-turn handling: each MT-Bench question has 2 turns. We batch-generate turn 1, build
turn-2 prompts that include the model's own turn-1 answer, then batch-generate turn 2.
Per-question sampling temperature follows fastchat.llm_judge.common.temperature_config.
"""
import argparse
import json
import os
import time
import uuid
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


TEMPERATURE_CONFIG = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}
DEFAULT_TEMP = 0.7


def load_questions(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_chat_prompt(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--answer-file", required=True)
    ap.add_argument("--max-new-token", type=int, default=1024)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--num-questions", type=int, default=None,
                    help="Process first N questions only (for smoke test)")
    args = ap.parse_args()

    questions = load_questions(args.question_file)
    if args.num_questions:
        questions = questions[: args.num_questions]
    print(f"[mtbench-gen] {args.model_id}: {len(questions)} questions, "
          f"TP={args.tensor_parallel_size}, max_len={args.max_model_len}",
          flush=True)

    answer_path = Path(args.answer_file)
    if answer_path.exists() and args.num_questions is None:
        with open(answer_path) as f:
            n_existing = sum(1 for line in f if line.strip())
        if n_existing >= len(questions):
            print(f"[mtbench-gen] SKIP: {n_existing} answers already exist in {answer_path}",
                  flush=True)
            return 0

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    turn1_prompts, turn1_params = [], []
    for q in questions:
        msgs = [{"role": "user", "content": q["turns"][0]}]
        turn1_prompts.append(build_chat_prompt(tokenizer, msgs))
        temp = TEMPERATURE_CONFIG.get(q["category"], DEFAULT_TEMP)
        turn1_params.append(SamplingParams(
            temperature=temp,
            top_p=1.0,
            max_tokens=args.max_new_token,
            seed=0,
        ))

    print(f"[mtbench-gen] turn 1: generating {len(turn1_prompts)} prompts...", flush=True)
    t0 = time.time()
    turn1_outputs = llm.generate(turn1_prompts, turn1_params)
    print(f"[mtbench-gen] turn 1: done in {time.time() - t0:.1f}s", flush=True)
    turn1_answers = [o.outputs[0].text.strip() for o in turn1_outputs]

    turn2_prompts, turn2_params = [], []
    for q, a1 in zip(questions, turn1_answers):
        msgs = [
            {"role": "user", "content": q["turns"][0]},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": q["turns"][1]},
        ]
        turn2_prompts.append(build_chat_prompt(tokenizer, msgs))
        temp = TEMPERATURE_CONFIG.get(q["category"], DEFAULT_TEMP)
        turn2_params.append(SamplingParams(
            temperature=temp,
            top_p=1.0,
            max_tokens=args.max_new_token,
            seed=0,
        ))

    print(f"[mtbench-gen] turn 2: generating {len(turn2_prompts)} prompts...", flush=True)
    t0 = time.time()
    turn2_outputs = llm.generate(turn2_prompts, turn2_params)
    print(f"[mtbench-gen] turn 2: done in {time.time() - t0:.1f}s", flush=True)
    turn2_answers = [o.outputs[0].text.strip() for o in turn2_outputs]

    answer_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = answer_path.with_suffix(answer_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for q, a1, a2 in zip(questions, turn1_answers, turn2_answers):
            rec = {
                "question_id": q["question_id"],
                "answer_id": uuid.uuid4().hex[:22],
                "model_id": args.model_id,
                "choices": [{"index": 0, "turns": [a1, a2]}],
                "tstamp": time.time(),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, answer_path)
    print(f"[mtbench-gen] wrote {len(questions)} answers to {answer_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
