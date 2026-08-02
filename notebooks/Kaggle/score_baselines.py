
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "rouge-score", "bert-score"], check=True)
print("packages installed")

import json, os
import numpy as np
from scipy import stats
from rouge_score import rouge_scorer
from bert_score import score as bertscore

D = "/kaggle/input/datasets/ayushsrivastava1805/baseline-scoring/baseline_scoring"
TEST_FILE = ("/kaggle/input/datasets/ayushsrivastava1805/python-train-cot/"
             "python_test_filtered.jsonl")

# The raw StarCoder file was uploaded with a .txt extension, so both names
# are tried. The contents are JSON either way.
SYSTEMS = [
    ("StarCoder2-3B zero-shot (raw)",
     [f"{D}/preds_baseline_starcoder.json", f"{D}/preds_baseline_starcoder.txt"]),
    ("StarCoder2-3B zero-shot (first line)",
     [f"{D}/preds_baseline_starcoder_firstline.json"]),
    ("Llama-3-8B-Instruct zero-shot",
     [f"{D}/preds_baseline_llama_extracted.json"]),
    ("Setup A (fine-tuned)",
     [f"{D}/preds_A_2ep.json"]),
    ("Setup B (fine-tuned)",
     [f"{D}/preds_B_comment_2ep.json"]),
]

test = [json.loads(l) for l in open(TEST_FILE)]
actuals = [s["msg"] for s in test]
print("test samples:", len(actuals))

clean = lambda p: p if str(p).strip() else "."
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def load_first(paths):
    for p in paths:
        if os.path.exists(p):
            return json.load(open(p))
    return None

scores, out = {}, {}
for name, paths in SYSTEMS:
    data = load_first(paths)
    if data is None:
        print(f"skipping {name}, file not found")
        continue
    preds = [clean(x) for x in data]
    if len(preds) != len(actuals):
        print(f"skipping {name}, has {len(preds)} rows but the test set has "
              f"{len(actuals)}")
        continue
    r = np.array([scorer.score(a, p)["rougeL"].fmeasure
                  for a, p in zip(actuals, preds)])
    _, _, F = bertscore(preds, actuals, lang="en", batch_size=64)
    b = F.numpy()
    words = float(np.mean([len(str(x).split()) for x in preds]))
    scores[name] = (r, b)
    out[name] = {"rougeL": float(r.mean()), "bertscore": float(b.mean()),
                 "mean_words": words}
    print(f"{name:<40} ROUGE-L {r.mean():.4f}   BERTScore {b.mean():.4f}"
          f"   words {words:.1f}")
    tag = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    np.save(f"bert_{tag}.npy", b)
    np.save(f"rouge_{tag}.npy", r)

# ----------------------------------------------------------- comparisons
KEY = "Setup A (fine-tuned)"
if KEY in scores:
    print("\npaired comparisons against Setup A")
    rA, bA = scores[KEY]
    for name, (r, b) in scores.items():
        if name == KEY:
            continue
        print(f"  {name}")
        print(f"    BERTScore  A {bA.mean():.4f} vs {b.mean():.4f}   "
              f"diff {bA.mean()-b.mean():+.4f}   "
              f"p = {stats.ttest_rel(bA, b).pvalue:.3g}")
        print(f"    ROUGE-L    A {rA.mean():.4f} vs {r.mean():.4f}   "
              f"diff {rA.mean()-r.mean():+.4f}   "
              f"p = {stats.ttest_rel(rA, r).pvalue:.3g}")

json.dump(out, open("baseline_results.json", "w"), indent=2)
print("\nsaved baseline_results.json")
