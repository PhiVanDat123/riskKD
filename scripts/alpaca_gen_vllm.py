"""AlpacaEval answer generation using vLLM offline batching.

Produces an alpaca-eval-compatible JSON list:
  [{"instruction": <str>, "output": <str>, "generator": <model_id>}, ...]

Loads the 805 AlpacaEval prompts via alpaca_eval.constants.get_alpaca_eval_data(),
applies the model's chat template, and batch-generates with vLLM. Greedy decoding
(temperature=0) per AlpacaEval convention.
"""
import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from alpaca_eval.constants import get_alpaca_eval_data


def build_chat_prompt(tokenizer, instruction):
    messages = [{"role": "user", "content": instruction}]
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
    ap.add_argument("--model-id", required=True, help="generator field for alpaca-eval")
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--max-new-token", type=int, default=2048)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--num-prompts", type=int, default=None,
                    help="Process first N prompts only (for smoke test)")
    args = ap.parse_args()

    ds = get_alpaca_eval_data()
    instructions = [ds[i]["instruction"] for i in range(len(ds))]
    if args.num_prompts:
        instructions = instructions[: args.num_prompts]
    print(f"[alpaca-gen] {args.model_id}: {len(instructions)} prompts, "
          f"TP={args.tensor_parallel_size}, max_len={args.max_model_len}",
          flush=True)

    output_path = Path(args.output_file)
    if output_path.exists() and args.num_prompts is None:
        with open(output_path) as f:
            existing = json.load(f)
        if isinstance(existing, list) and len(existing) >= len(instructions):
            print(f"[alpaca-gen] SKIP: {len(existing)} outputs already in {output_path}",
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

    prompts = [build_chat_prompt(tokenizer, ins) for ins in instructions]
    sampling = SamplingParams(
        temperature=0.0,  # greedy per AlpacaEval convention
        top_p=1.0,
        max_tokens=args.max_new_token,
        seed=0,
    )
    import time
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    print(f"[alpaca-gen] generated {len(outputs)} answers in {time.time() - t0:.1f}s",
          flush=True)

    records = []
    for ins, out in zip(instructions, outputs):
        records.append({
            "instruction": ins,
            "output": out.outputs[0].text.strip(),
            "generator": args.model_id,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)
    print(f"[alpaca-gen] wrote {len(records)} records to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
