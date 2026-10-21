"""Text identity, membership, and complementary shadow-set construction.

Two texts are treated as the same candidate if their whitespace-normalized strings match
or if the token ids of their scored prefix (first ``max_length`` tokens) match exactly.
Overlapping candidates are merged into groups with union-find, and groups (not single
texts) are assigned to the IN side of exactly one model in each complementary pair.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def record_text(r) -> str:
    return r if isinstance(r, str) else r["text"]


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def prefix_keys(tokenizer, texts: Sequence[str], max_length: int = 512, chunk: int = 256) -> list[str]:
    """SHA-256 of the scored token prefix of each text (no special tokens, truncated)."""
    keys = []
    for s in range(0, len(texts), chunk):
        enc = tokenizer(list(texts[s:s + chunk]), add_special_tokens=False, truncation=True,
                        max_length=max_length, padding=False)
        for ids in enc["input_ids"]:
            keys.append(hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest())
    return keys


class CandidateIndex:
    """Normalized-text and prefix keys of candidate sets (e.g. {"dev": [...], "val": [...]})."""

    def __init__(self, tokenizer, candidate_sets: dict[str, Sequence[str]], max_length: int = 512):
        self.tokenizer, self.max_length = tokenizer, max_length
        self.names = list(candidate_sets)
        self.norm = {k: [normalize_text(t) for t in v] for k, v in candidate_sets.items()}
        self.pref = {k: prefix_keys(tokenizer, v, max_length) for k, v in candidate_sets.items()}

    def membership(self, train_texts: Iterable[str]) -> dict[str, np.ndarray]:
        """Boolean IN flag per candidate: exact normalized match OR identical scored prefix."""
        train_texts = list(train_texts)
        ns = {normalize_text(t) for t in train_texts}
        ps = set(prefix_keys(self.tokenizer, train_texts, self.max_length))
        return {k: np.array([(a in ns) or (b in ps) for a, b in zip(self.norm[k], self.pref[k])])
                for k in self.names}

    def items(self) -> list[tuple[str, int]]:
        return [(k, i) for k in self.names for i in range(len(self.norm[k]))]

    def groups(self) -> dict[int, list[int]]:
        """Union-find over all candidates: linked by equal normalized text or equal prefix key."""
        items = self.items()
        parent = list(range(len(items)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        first = {}
        for idx, (k, i) in enumerate(items):
            for key in (("n", self.norm[k][i]), ("p", self.pref[k][i])):
                if key in first:
                    parent[find(idx)] = find(first[key])
                else:
                    first[key] = idx
        out: dict[int, list[int]] = {}
        for idx in range(len(items)):
            out.setdefault(find(idx), []).append(idx)
        return out

    def excludes(self, texts: Sequence[str]) -> list[bool]:
        """True where a text collides with any candidate (use to filter auxiliary training data)."""
        ns = {t for v in self.norm.values() for t in v}
        ps = {p for v in self.pref.values() for p in v}
        keys = prefix_keys(self.tokenizer, texts, self.max_length)
        return [normalize_text(t) in ns or k in ps for t, k in zip(texts, keys)]


def assign_complementary(groups: dict[int, list[int]], seed: int) -> tuple[list[int], list[int]]:
    """Split candidate groups into two balanced sides; side s is IN for model s of the pair.

    Groups are visited in a seeded random order and each goes to the currently smaller
    side (random tie-break), so every candidate is IN exactly one of the two models.
    """
    rng = np.random.default_rng(seed)
    sides: tuple[list[int], list[int]] = ([], [])
    sizes = [0, 0]
    for g in rng.permutation(sorted(groups)):
        g = int(g)
        s = int(rng.integers(2)) if sizes[0] == sizes[1] else int(sizes[1] < sizes[0])
        sides[s].extend(groups[g])
        sizes[s] += len(groups[g])
    return sides


def build_shadow_set(in_texts: Sequence[str], aux_pool: Sequence[str], size: int,
                     sample_seed: int, shuffle_seed: int) -> list[str]:
    """IN candidates plus auxiliary texts (already filtered against all candidates) up to ``size``."""
    import random

    ev = list(dict.fromkeys(in_texts))
    n_aux = size - len(ev)
    if n_aux < 0 or len(aux_pool) < n_aux:
        raise ValueError(f"need {n_aux} auxiliary texts, have {len(aux_pool)}")
    texts = ev + random.Random(sample_seed).sample(list(aux_pool), n_aux)
    random.Random(shuffle_seed).shuffle(texts)
    return texts
