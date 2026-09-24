# Membership Inference Attacks on a Fine-Tuned GPT-2

Can you tell whether a piece of text was in a language model's fine-tuning data, using only the model's outputs? This project implements and compares three reference-free membership inference attacks (MIAs) against a GPT-2 model fine-tuned on WikiText-103, and measures how much membership actually leaks, especially at the low false-positive rates that matter for a privacy audit.

Built as a project for Foundations of Privacy at CMU (Fall 2026).

## Threat model

- The attacker has black-box access to the target model plus its logits, and a candidate text.
- No shadow models, no reference models, no retraining. Every score comes from a single forward pass.
- Labels are used only after scoring, to evaluate the attacks.

## Attacks

All scores follow one convention: **higher score = more likely a member**.

| Attack | Score | Idea |
|---|---|---|
| **Average loss** | negative mean token cross-entropy | Training exposure makes member texts more predictable. |
| **Min-K%** ([Shi et al., 2024](https://arxiv.org/abs/2310.16789)) | negative mean of the K% highest token losses | Even a member's hardest tokens should be less surprising than a non-member's. |
| **zlib-normalized loss** ([Carlini et al., 2021](https://arxiv.org/abs/2012.07805)) | mean loss divided by zlib-compressed byte length | Discount texts that are easy for reasons unrelated to memorization (repetition, boilerplate). |

Implementation details: next-token alignment (drop the last logit and the first target), per-token losses with `reduction="none"`, padding masked out of both the sum and the count, inputs truncated at 256 tokens. Min-K% at K = 100% is checked to reproduce the average-loss scores exactly (`np.allclose`, atol 1e-5).

## Results

2,000 texts (1,000 members, 1,000 non-members).

| Attack | AUC | TPR @ FPR < 1% | TPR @ FPR < 5% |
|---|---|---|---|
| Average loss | 0.7136 | 3.3% | 19.5% |
| Min-K 5% | 0.7050 | 5.4% | 17.0% |
| **Min-K 10%** | 0.7088 | **5.9%** | 16.7% |
| Min-K 20% | 0.7085 | 4.5% | 18.1% |
| Min-K 40% | 0.7108 | 3.2% | 19.1% |
| **Min-K 60%** | **0.7153** | 3.0% | 18.9% |
| Min-K 80% | 0.7143 | 3.6% | **19.6%** |
| zlib | 0.5929 | 5.4% | 14.2% |

<p align="center">
  <img src="figures/roc_comparison.png" width="90%">
</p>

**Takeaways**

- **Membership leaks, but weakly where it matters.** Average loss reaches AUC 0.71, well above chance, yet at FPR < 1% it identifies only 33 of 1,000 members.
- **Min-K% moves performance between regimes instead of improving it overall.** Small K (10%) nearly doubles detection at FPR < 1% (33 to 59 members) but lowers AUC. Large K (60%) gives the best AUC, only 0.0017 above plain loss.
- **This zlib normalization hurts.** Dividing by compressed byte length appears to add more length sensitivity than it removes, dropping AUC to 0.59.
- **No single attack wins on all three metrics.**

## Limitations

- K was chosen on the same labeled set used to report results, so the table is in-sample. Differences of a few thousandths of AUC should be treated as noise.
- No confidence intervals.
- Texts longer than 256 tokens are judged on their prefix only.
- Low loss signals predictability, not membership. Easy non-members are the main source of false positives, which a reference-model attack (e.g. LiRA-style calibration) would address.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook membership_inference.ipynb
```

The target model and evaluation data are not included in this repo. The notebook expects:

```
models/gpt2_target/     fine-tuned GPT-2 checkpoint + tokenizer (Hugging Face format)
data/test.json          list of 2,000 texts
data/test_label.json    list of 0/1 membership labels
```

Runs on CPU, Apple Silicon (MPS) or CUDA. On a single L4 GPU the full evaluation takes a few minutes.

## Repository

```
membership_inference.ipynb   all three attacks, K sweep, evaluation and plots (with saved outputs)
figures/                     ROC curves
requirements.txt
```

## References

- Shi et al. (2024). *Detecting Pretraining Data from Large Language Models.* ICLR.
- Carlini et al. (2021). *Extracting Training Data from Large Language Models.* USENIX Security.
- Merity et al. (2016). *Pointer Sentinel Mixture Models* (WikiText-103).
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2).
