
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# !pip install -q unsloth
# !pip install -q rouge-score bert-score

import json, random
import numpy as np
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel


N_SAMPLES  = 1233      
MAX_NEW    = 150
TEST_FILE  = "/kaggle/input/datasets/ayushsrivastava1805/python-train-cot/python_test_filtered.jsonl"
SEED       = 42

test = [json.loads(l) for l in open(TEST_FILE)]
random.seed(SEED)
if N_SAMPLES < len(test):
    idx = sorted(random.sample(range(len(test)), N_SAMPLES))
    test = [test[i] for i in idx]
else:
    idx = list(range(len(test)))
print("evaluating on", len(test), "samples")
json.dump(idx, open("baseline_sample_idx.json", "w"))

# ----------------------------------------------------------------- helpers
def strip_license_header(code):
    lines = str(code).split("\n")
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("#") or s.startswith('"""') or s.startswith("'''") or s == "":
            i += 1
        else:
            break
    return "\n".join(lines[i:]) if i < len(lines) else str(code)

def build_context(s):
    oldf  = strip_license_header(s.get("oldf", ""))[:1500]
    patch = s.get("patch", "")[:800]
    return oldf, patch

# Same prompt shape the fine-tuned models saw, so the comparison is fair.
def prompt_starcoder(s):
    oldf, patch = build_context(s)
    return f"### Code:\n{oldf}\n\n### Change:\n{patch}\n\n### Review Comment:\n"

# Llama is instruction tuned, so it gets an instruction instead.
def prompt_llama(s, tok):
    oldf, patch = build_context(s)
    user = (
        "You are an experienced code reviewer. Read the code change below "
        "and write a single short review comment, the way a human reviewer "
        "would leave it on a pull request. Write only the comment itself, "
        "with no preamble and no explanation.\n\n"
        f"Code:\n{oldf}\n\nChange:\n{patch}"
    )
    msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def generate_all(model, tok, prompts, max_new=MAX_NEW):
    out = []
    for p in tqdm(prompts):
        inputs = tok(p, return_tensors="pt", truncation=True,
                     max_length=1280).to("cuda")
        with torch.no_grad():
            g = model.generate(**inputs, max_new_tokens=max_new,
                               temperature=0.7, do_sample=True,
                               pad_token_id=tok.eos_token_id)
        full = tok.decode(g[0], skip_special_tokens=True)
        out.append(full[len(p):].strip() if full.startswith(p)
                   else full.split(p[-80:])[-1].strip())
    return out

# ----------------------------------------------------------------- baseline 1
print("\n=== baseline 1: base StarCoder2-3B, no fine-tuning ===")
model, tok = FastLanguageModel.from_pretrained(
    model_name="bigcode/starcoder2-3b",
    max_seq_length=1280, dtype=None, load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
preds_sc = generate_all(model, tok, [prompt_starcoder(s) for s in test])
json.dump(preds_sc, open("preds_baseline_starcoder.json", "w"))
del model, tok
torch.cuda.empty_cache()

# ----------------------------------------------------------------- baseline 2
print("\n=== baseline 2: zero-shot Llama-3-8B-Instruct ===")
model, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length=2048, dtype=None, load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
preds_ll = generate_all(model, tok, [prompt_llama(s, tok) for s in test])
json.dump(preds_ll, open("preds_baseline_llama.json", "w"))
del model, tok
torch.cuda.empty_cache()

# ----------------------------------------------------------------- scoring
print("\n=== scoring ===")
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from scipy import stats

actuals = [s["msg"] for s in test]
clean = lambda p: p if str(p).strip() else "."

def evaluate(preds, name):
    p = [clean(x) for x in preds]
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    r = np.array([sc.score(a, q)["rougeL"].fmeasure for a, q in zip(actuals, p)])
    _, _, F = bertscore(p, actuals, lang="en", batch_size=64)
    b = F.numpy()
    print(f"{name:<28} ROUGE-L {r.mean():.3f}   BERTScore {b.mean():.3f}")
    return r, b

r_sc, b_sc = evaluate(preds_sc, "StarCoder2-3B zero-shot")
r_ll, b_ll = evaluate(preds_ll, "Llama-3-8B-Instruct zero-shot")

#
# preds_A_full = json.load(open("/kaggle/input/<your-preds-dataset>/preds_A.json"))
# preds_A = [preds_A_full[i] for i in idx]
# r_A, b_A = evaluate(preds_A, "Setup A (fine-tuned)")
# print("\nfine-tuning gain over base StarCoder2:")
# print("  BERTScore diff", f"{b_A.mean() - b_sc.mean():+.3f}",
#       "p =", f"{stats.ttest_rel(b_A, b_sc).pvalue:.4g}")
# print("  ROUGE-L   diff", f"{r_A.mean() - r_sc.mean():+.3f}",
#       "p =", f"{stats.ttest_rel(r_A, r_sc).pvalue:.4g}")

json.dump({
    "n_samples": len(test),
    "starcoder_zeroshot": {"rougeL": float(r_sc.mean()), "bertscore": float(b_sc.mean())},
    "llama_zeroshot":     {"rougeL": float(r_ll.mean()), "bertscore": float(b_ll.mean())},
}, open("baseline_results.json", "w"), indent=2)
print("\nsaved baseline_results.json")