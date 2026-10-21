"""Membership inference attacks against LoRA fine-tuned causal language models.

Modules
-------
data     : text normalization, exact/prefix membership, overlap groups, complementary IN/OUT splits
losses   : per-token loss extraction for base + LoRA adapters
attacks  : reference-model scores (Gaussian likelihood ratio, mean-gap, RMIA, EZ, shared-variance LR)
metrics  : strict TPR at low FPR, AUC, stratified paired bootstrap
combine  : tail-focused pairwise ranking combiner
"""

from .attacks import (
    gaussian_lr,
    mean_gap,
    rmia_log_ratio,
    shared_variance_lr,
    ez_error_zone_ratio,
)
from .metrics import tpr_at_fpr, strict_tpr_fast, stratified_bootstrap, paired_delta

__all__ = [
    "gaussian_lr",
    "mean_gap",
    "rmia_log_ratio",
    "shared_variance_lr",
    "ez_error_zone_ratio",
    "tpr_at_fpr",
    "strict_tpr_fast",
    "stratified_bootstrap",
    "paired_delta",
]
