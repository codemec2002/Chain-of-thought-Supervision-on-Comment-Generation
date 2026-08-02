

import json, re, difflib, os, random
from collections import Counter
import numpy as np
from scipy import stats

JUDGE_ITEMS = "judge_items_full.json"
RUNS = [
    ("run 2 (1.6 epoch, 22k)", "preds_A_2ep.json", "preds_B_comment_2ep.json",
     "bert_A_2ep.npy", "bert_B_2ep.npy", "rouge_A_2ep.npy", "rouge_B_2ep.npy"),
    ("run 3 (2.1 epoch, 15k)", "preds_A_ck4000.json", "preds_B_comment_ck4000.json",
     "bert_A_ck4000.npy", "bert_B_ck4000.npy", "rouge_A_ck4000.npy", "rouge_B_ck4000.npy"),
]
ORDER = ["control flow", "rewrite", "new code", "imports and style", "documentation"]
MIN_N = 25

# ------------------------------------------------------------ diff parsing
def parse(patch):
    add, rem = [], []
    for ln in str(patch).split("\n"):
        if ln.startswith("+++") or ln.startswith("---"):
            continue
        if ln.startswith("+"):
            add.append(ln[1:])
        elif ln.startswith("-"):
            rem.append(ln[1:])
    return add, rem

def is_comment(l):
    return l.strip().startswith(("#", '"""', "'''", "*", "//", "/*"))

def is_blank(l):
    return l.strip() == ""

def is_import(l):
    return l.strip().startswith(("import ", "from "))

# A change counts as control flow only if it introduces a real branch or an
# error path. Bare return, continue and break were deliberately left out:
# almost every Python function contains a return, so including it pulled in
# ordinary new functions that have no conditional logic at all.
GUARD = re.compile(r"(^|\W)(if|elif|else|try|except|finally|raise|assert|"
                   r"with\s+\w+)(\W|$)")
NONE_CHK = re.compile(r"(is\s+None|is\s+not\s+None|!=\s*None|==\s*None|"
                      r"isinstance\(|hasattr\()")

def similar_frac(add, rem):
    """Fraction of added lines that closely match a removed line."""
    if not add or not rem:
        return 0.0
    hits, pool = 0, list(rem)
    for a in add:
        best, bi = 0.0, -1
        for i, r in enumerate(pool):
            ratio = difflib.SequenceMatcher(None, a.strip(), r.strip()).ratio()
            if ratio > best:
                best, bi = ratio, i
        if best > 0.6:
            hits += 1
            if bi >= 0:
                pool.pop(bi)
    return hits / len(add)

def categorise(patch):
    add, rem = parse(patch)
    code_a = [l for l in add if not is_comment(l) and not is_blank(l)]
    code_r = [l for l in rem if not is_comment(l) and not is_blank(l)]
    if not code_a and not code_r:
        return "documentation"
    if code_a and all(is_import(l) for l in code_a + code_r):
        return "imports and style"
    if not code_a and code_r:
        return "rewrite"
    if similar_frac(code_a, code_r) > 0.5 and code_r:
        return "rewrite"
    if any(GUARD.search(l) or NONE_CHK.search(l) for l in code_a):
        return "control flow"
    return "new code"

# ------------------------------------------------------------ load
items = json.load(open(JUDGE_ITEMS))
actuals = [it["actual"] for it in items]
cats = np.array([categorise(it["patch"]) for it in items])

counts = Counter(cats)
print("category distribution, n =", len(cats))
for k in ORDER:
    print(f"  {k:<20}{counts[k]:>5}  {100*counts[k]/len(cats):.1f}%")

len_ref = np.array([len(str(x).split()) for x in actuals])
results = {"distribution": {k: int(counts[k]) for k in ORDER}, "runs": {}}

# ------------------------------------------------------------ per run
for label, fa, fb, ba_f, bb_f, ra_f, rb_f in RUNS:
    if not os.path.exists(fa):
        print("\nskipping", label, "files not found")
        continue
    pA = json.load(open(fa)); pB = json.load(open(fb))
    bA = np.load(ba_f); bB = np.load(bb_f)
    rA = np.load(ra_f); rB = np.load(rb_f)
    lA = np.array([len(str(x).split()) for x in pA])
    lB = np.array([len(str(x).split()) for x in pB])

    print("\n" + "=" * 78)
    print(label)
    print(f"{'category':<20}{'n':>5}{'BERT A':>9}{'BERT B':>9}{'diff':>9}"
          f"{'p':>10}{'ROUGE A':>10}{'ROUGE B':>10}{'len A':>8}{'len B':>8}")
    run_rows = []
    for k in ORDER:
        idx = np.where(cats == k)[0]
        if len(idx) < MIN_N:
            continue
        t_b = stats.ttest_rel(bB[idx], bA[idx])
        t_r = stats.ttest_rel(rB[idx], rA[idx])
        print(f"{k:<20}{len(idx):>5}{bA[idx].mean():>9.4f}{bB[idx].mean():>9.4f}"
              f"{bB[idx].mean()-bA[idx].mean():>+9.4f}{t_b.pvalue:>10.4f}"
              f"{rA[idx].mean():>10.4f}{rB[idx].mean():>10.4f}"
              f"{lA[idx].mean():>8.1f}{lB[idx].mean():>8.1f}")
        run_rows.append({
            "category": k, "n": int(len(idx)),
            "bert_A": float(bA[idx].mean()), "bert_B": float(bB[idx].mean()),
            "bert_diff": float(bB[idx].mean() - bA[idx].mean()),
            "bert_p": float(t_b.pvalue),
            "rouge_A": float(rA[idx].mean()), "rouge_B": float(rB[idx].mean()),
            "rouge_diff": float(rB[idx].mean() - rA[idx].mean()),
            "rouge_p": float(t_r.pvalue),
            "len_A": float(lA[idx].mean()), "len_B": float(lB[idx].mean()),
            "len_ref": float(len_ref[idx].mean()),
        })

    thresh = 0.05 / len(run_rows)
    sig = [r["category"] for r in run_rows if r["bert_p"] < thresh]
    print(f"  Bonferroni threshold {thresh:.4f}, significant on BERTScore: "
          f"{sig if sig else 'none'}")

    # ------------------------------------------------------ length confound
    d_bert, d_len = bB - bA, lB - lA
    r_corr, p_corr = stats.pearsonr(d_len, d_bert)
    shorter, longer = lB <= lA, lB > lA
    t_short = stats.ttest_rel(bB[shorter], bA[shorter])
    t_long = stats.ttest_rel(bB[longer], bA[longer])
    print("  length confound check")
    print(f"    corr(length diff, BERTScore diff) r={r_corr:.3f} p={p_corr:.2e}")
    print(f"    B not longer  n={shorter.sum():>5}  "
          f"diff={bB[shorter].mean()-bA[shorter].mean():+.4f} p={t_short.pvalue:.4f}")
    print(f"    B longer      n={longer.sum():>5}  "
          f"diff={bB[longer].mean()-bA[longer].mean():+.4f} p={t_long.pvalue:.4f}")

    results["runs"][label] = {
        "categories": run_rows,
        "bonferroni_threshold": float(thresh),
        "length_confound": {
            "pearson_r": float(r_corr), "pearson_p": float(p_corr),
            "b_not_longer_n": int(shorter.sum()),
            "b_not_longer_diff": float(bB[shorter].mean() - bA[shorter].mean()),
            "b_not_longer_p": float(t_short.pvalue),
            "b_longer_n": int(longer.sum()),
            "b_longer_diff": float(bB[longer].mean() - bA[longer].mean()),
            "b_longer_p": float(t_long.pvalue),
        },
    }

json.dump(results, open("categorical_results.json", "w"), indent=2)
print("\nsaved categorical_results.json")

# ------------------------------------------------------------ manual check
# The proposal promised a manual validation of the rule based classifier.
# This writes 100 random samples so the assignments can be checked by hand
# and the accuracy reported in the thesis.
random.seed(42)
pick = random.sample(range(len(items)), 100)
json.dump([{"idx": int(i), "assigned_category": str(cats[i]),
            "patch": items[i]["patch"][:600], "reference_comment": actuals[i]}
           for i in pick], open("category_check_100.json", "w"), indent=2)
print("saved category_check_100.json for manual validation")
