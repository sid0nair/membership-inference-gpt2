"""Per-token losses of a base model with a LoRA adapter.

Settings match every cached score used in the experiments: fp16 forward on CUDA, cross-entropy
on fp32 logits, batch size 2, right padding with pad = eos, truncation at 512 tokens, and the
scored positions are the attention-masked next-token predictions (tokens 2..T).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def load_lora(base_model_name: str, adapter_dir, tokenizer, device="cuda", dtype=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = dtype or (torch.float16 if str(device).startswith("cuda") else torch.float32)
    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model.to(device).eval()


def per_token_losses(model, tokenizer, texts: Sequence[str], batch_size: int = 2,
                     max_length: int = 512, device="cuda") -> list[np.ndarray]:
    import torch
    import torch.nn.functional as F

    out: list[np.ndarray] = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch_size):
            b = tokenizer(list(texts[s:s + batch_size]), return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length)
            b = {k: v.to(device) for k, v in b.items()}
            tgt, mask = b["input_ids"][:, 1:], b["attention_mask"][:, 1:].bool()
            logits = model(**b, use_cache=False).logits[:, :-1, :]
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                   reduction="none").reshape(tgt.shape)
            out += [row[m].float().cpu().numpy() for row, m in zip(loss, mask)]
    return out


def save_losses(path, losses: Sequence[np.ndarray]) -> None:
    np.savez_compressed(path, lengths=np.asarray([a.size for a in losses]), losses=np.concatenate(losses))


def load_losses(path, key: str = "losses") -> list[np.ndarray]:
    z = np.load(Path(path))
    return [a.astype(np.float64) for a in np.split(z[key], np.cumsum(z["lengths"])[:-1])]


def mean_losses(losses: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([a.mean() for a in losses], dtype=np.float64)
