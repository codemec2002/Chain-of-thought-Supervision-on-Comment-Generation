# Chain-of-Thought Supervision for Code Review Comment Generation

MSc Artificial Intelligence research project, National College of Ireland.

**Author:** Ayush Srivastava (24268798) | **Supervisor:** Dr. Sobia Bano

Does teaching a small code model to reason before it writes a review comment make the comment better? This repo fine-tunes StarCoder2-3B in two ways on the same data and compares the outputs.

| Setup | Training target |
|---|---|
| **A - Direct** | code change -> review comment |
| **B - Chain-of-thought** | code change -> reasoning steps -> review comment |

Everything else is kept identical: base model, hyperparameters, sequence length, seed and samples.

## Key results

Evaluated on 1,233 held-out Python samples from CodeReviewer.

| System | ROUGE-L | BERTScore F1 |
|---|---|---|
| StarCoder2-3B zero-shot (raw) | 0.0516 | 0.7796 |
| StarCoder2-3B zero-shot (first line) | 0.0401 | 0.8106 |
| Llama-3-8B-Instruct zero-shot | 0.1041 | 0.8498 |
| Setup A (direct, 1 epoch) | 0.108 | 0.851 |
| **Setup B (CoT, 1 epoch)** | **0.109** | **0.854** |

- **BERTScore:** Setup B is significantly better in all three training runs (p < 0.0001, p < 0.0001, p = 0.004). The effect is small, about +0.003.
- **ROUGE-L:** no significant difference in any run.
- **LLM judge (Llama-3-8B-Instruct, full test set):** B preferred 631 times (51.2%), A 576 (46.7%), tie 26 (2.1%), p = 0.06, not significant.
- **Human validation (50 samples):** A 26, B 24. Agreement with the LLM judge is 63%, Cohen's kappa 0.27 (fair).

In short: CoT supervision gives a small but reproducible gain in meaning, not in exact wording. The model says the right thing in different words.

## Method

**Dataset.** CodeReviewer (Li et al., 2022), from Zenodo (DOI [10.5281/zenodo.6900648](https://doi.org/10.5281/zenodo.6900648)). Fields used are `patch`, `oldf` and `msg`. The training file has no language field, so a TF-IDF character n-gram + LinearSVC classifier is trained to recover it, and only Python samples are kept.

**Filtering.** Comments under five words, generic approvals (lgtm, ship it), very large files and exact duplicates are removed.

| Split | Samples |
|---|---|
| Train, Setup A | 22,389 |
| Train, Setup B | 22,286 |
| Validation | 1,243 |
| Test | 1,233 |

**Reasoning labels.** For Setup B, GPT-4o-mini writes 2 to 4 short reasoning steps that lead to the real reviewer comment. Labels that fail the quality checks are dropped (103 of 22,389).

**Training.** StarCoder2-3B with QLoRA through Unsloth, on a single Kaggle Tesla T4.

| Parameter | Value |
|---|---|
| Base model | `bigcode/starcoder2-3b`, 4-bit |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable parameters | 9,093,120 (0.30%) |
| Learning rate | 2e-4, cosine schedule, 50 warmup steps |
| Batch size | 2 x 4 gradient accumulation (effective 8) |
| Optimizer | adamw_8bit |
| Max sequence length | 1280 |
| Max new tokens at inference | 150 |
| Seed | 42 |

**Training runs.** Each run trains both setups.

| Run | Data | Epochs |
|---|---|---|
| 1 | full training set | 1 |
| 2 | full training set | about 1.6 |
| 3 | 15k subset (same ids for A and B) | about 2.1 |

## Repository structure

```
notebooks/
  01_data_exploration.ipynb     first look at CodeReviewer
  data_exploration.ipynb        exploration for the comment generation task
  02_data_preprocessing.ipynb   language classifier, filtering, splits
  03_cot_labels.ipynb           reasoning labels with GPT-4o-mini
  categorical_analysis.py       results broken down by type of code change
  figs/                         dataset figures
  Kaggle/
    thesis-2.ipynb                      training, full data (SETUP = "A" or "B")
    thesis_3epochs.ipynb                training, 15k subset
    thesis-evaluation-2.ipynb           ROUGE-L and BERTScore, 1 epoch adapters
    thesis-3.ipynb                      ROUGE-L and BERTScore, later checkpoints
    full-llm-as-judge.ipynb             LLM judge on the full test set
    llm-as-judge-after-3-epochs.ipynb   LLM judge for the 15k subset run
    baseline_eval.py                    zero-shot baseline predictions
    score_baselines.py                  scores the baselines
```

## How to run

The local notebooks (01 to 03) run on CPU with Python 3.12. The `Kaggle/` files need a GPU and are written for Kaggle with a T4.

1. **Data.** Download CodeReviewer from Zenodo and run `02_data_preprocessing.ipynb` to get the filtered Python splits.
2. **Reasoning labels.** Put `OPENAI_API_KEY` in a `.env` file and run `03_cot_labels.ipynb`.
3. **Train.** Upload the splits as a Kaggle dataset, open `Kaggle/thesis-2.ipynb`, set `SETUP = "A"` or `"B"`, and Run All. One setup per run.
4. **Evaluate.** Point `ADAPTER_A` and `ADAPTER_B` at the saved adapters in the evaluation notebook, then run the LLM judge notebook.
5. **Baselines and breakdown.** Run `baseline_eval.py`, `score_baselines.py` and `categorical_analysis.py`.

Notes:
- Install only `unsloth` on Kaggle. It pins compatible versions of transformers, peft, trl and bitsandbytes. Installing those separately breaks the import.
- The notebooks set `CUDA_VISIBLE_DEVICES=0` so that only one GPU is used on a T4 x2 session.
- Dataset paths inside the Kaggle files point to my own Kaggle datasets. Change them to yours.

## Limitations

- The gain is small and shows up only on the semantic metric.
- One base model (3B) and one language (Python).
- Reasoning labels are generated by a teacher model with the final comment visible, so they are reconstructed, not real reviewer reasoning.
- Human validation is 50 samples by a single annotator.

## References

- Li et al. (2022). Automating Code Review Activities by Large-Scale Pre-training. ESEC/FSE.
- Lozhkov et al. (2024). StarCoder 2 and The Stack v2: The Next Generation.
