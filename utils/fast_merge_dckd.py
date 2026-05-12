"""Zero-copy column-wise merge of per-split teacher-logp datasets into the *-dckd dataset.
concatenate_datasets(..., axis=1) instead of a Python-list->pyarrow round-trip. Atomic via .tmp rename."""
import os, shutil, argparse
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
try:
    base = load_dataset(a.base)
except Exception:
    base = load_from_disk(a.base)
print("[fast_merge] base splits:", list(base.keys()))
result = {}
for split in [s for s in ("train", "test") if s in base]:
    cp = os.path.join(a.root, f"{a.tag}-chosen-logp-{split}")
    rp = os.path.join(a.root, f"{a.tag}-rejected-logp-{split}")
    if not (os.path.isdir(cp) and os.path.isdir(rp)):
        print(f"[fast_merge] WARN missing logp dir for split={split}, skip"); continue
    c = load_from_disk(cp); r = load_from_disk(rp); b = base[split]
    assert len(b) == len(c) == len(r), f"row mismatch {split}: base={len(b)} chosen={len(c)} rejected={len(r)}"
    c2 = c.select_columns(["chosen_compressed_probs", "chosen_labels"])
    r2 = r.select_columns(["rejected_compressed_probs", "rejected_labels"])
    merged = concatenate_datasets([b.flatten_indices(), c2, r2], axis=1)
    print(f"[fast_merge] {split}: cols={merged.column_names} n={len(merged)}")
    result[split] = merged
dd = DatasetDict(result)
tmp = a.out + ".tmp"
if os.path.exists(tmp): shutil.rmtree(tmp)
print(f"[fast_merge] writing -> {tmp}"); dd.save_to_disk(tmp)
if os.path.exists(a.out): shutil.rmtree(a.out)
os.rename(tmp, a.out); print(f"[fast_merge] DONE -> {a.out}")
