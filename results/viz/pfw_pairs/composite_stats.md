# PFW per-pair stats (24 UltraFeedback examples each)

| Pair | Trust  | Frontier | ErrorGate | p10  | p50  | p90  | Gini-proxy |
|------|-------:|---------:|----------:|-----:|-----:|-----:|-----------:|
| Qwen3-8B  -->  Qwen3-1.7B | 0.743 | 0.542 | 1.238 | 0.001 | 1.130 | 1.525 | 0.269 |
| Qwen3-1.7B  -->  Qwen3-0.6B | 0.745 | 0.555 | 1.239 | 0.012 | 1.112 | 1.511 | 0.265 |
| Llama-3.1-8B  -->  Llama-3.2-1B | 0.657 | 0.597 | 1.240 | 0.008 | 1.098 | 1.577 | 0.296 |

**Observations.**
- ErrorGate ~ 1.24 across all 3 pairs - the loss-cap step has the same activation rate regardless of student size or family.
- Frontier ~ 0.55-0.60: roughly half the tokens fall in the teachable band; this fraction is stable across pairs.
- Trust drops from 0.74 (Qwen) to 0.66 (Llama) - Llama student is less aligned with its DPO teacher's preferences.
- Gini-proxy 0.27-0.30: PFW concentrates noticeably more than kl_inv (which has Gini ~ 0.10-0.15) and uniform (0).
- Bottom 10% of tokens receive ~0.001-0.01 weight: PFW effectively zeros out non-frontier tokens.
- Top 10% of tokens receive 1.5-1.6x uniform: budget is reallocated, not just rescaled.
