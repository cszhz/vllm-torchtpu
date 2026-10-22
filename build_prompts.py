"""Real-text decode contexts for p3_d_leg_bench --prompt-file: tokenize the image's Python sources
(deterministic sorted file order) with the model tokenizer and cut N records of L tokens each.
usage: build_prompts.py <tokenizer_dir> <N> <L> <out.jsonl>"""
import json, os, sys
from transformers import AutoTokenizer
tokdir, n, L, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
tok = AutoTokenizer.from_pretrained(tokdir, trust_remote_code=True)
root = "/usr/local/lib/python3.12/site-packages"
files = []
for d, _, fs in os.walk(root):
    for f in fs:
        if f.endswith(".py"):
            files.append(os.path.join(d, f))
files.sort()
ids, recs = [], []
for f in files:
    try:
        text = open(f, errors="ignore").read()
    except OSError:
        continue
    ids.extend(tok(text, add_special_tokens=False)["input_ids"])
    while len(ids) >= L and len(recs) < n:
        recs.append(ids[:L]); ids = ids[L:]
    if len(recs) >= n:
        break
with open(out, "w") as fo:
    for r in recs:
        fo.write(json.dumps({"prefix": r}) + "\n")
print(f"BD_PROMPTS {len(recs)} records x {L} tokens from {len(files)} files -> {out}", flush=True)
assert len(recs) == n, "not enough text"
