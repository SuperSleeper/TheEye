"""Name animals more precisely than the detector can, from the CLIP fingerprint each track already has.

The detector (COCO classes) only knows bird / cat / dog, so a squirrel comes out as "bird". CLIP compares the
track's picture with short descriptions, in two steps:
  1. which animal it is (config animals.kinds), which corrects squirrel-as-bird;
  2. for birds / dogs / cats, which kind (animals.birds / dogs / cats), e.g. crow or pomeranian.
Only a confident answer is used; otherwise the detector's label stands. No extra image work: the picture's
fingerprint was already made for visual search, and comparing it with ~50 cached text fingerprints is a tiny
matrix product.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

from .config import AnimalsConfig

log = logging.getLogger("theeye.animals")

ANIMAL_LABELS = {"bird", "cat", "dog"}           # detector labels that get a closer look
TEMPLATES = ["a photo of a {}.", "a security camera photo of a {}.", "a blurry low-resolution photo of a {}.",
             "a photo of a {} outdoors."]
SCALE = 100.0                                    # CLIP's logit scale: turns cosine similarities into odds


class SubtypeClassifier:
    def __init__(self, cfg: AnimalsConfig, embedder):
        self.cfg = cfg
        self.embedder = embedder
        self._cache: dict[tuple, np.ndarray] = {}
        self._lock = threading.Lock()

    def fine_list(self, kind: str) -> list[str]:
        return {"bird": self.cfg.birds, "dog": self.cfg.dogs, "cat": self.cfg.cats}.get(kind, [])

    def names(self) -> set[str]:  # noqa: D401
        """Every name this can produce, lower case (for search)."""
        return {n.strip().lower() for n in (*self.cfg.kinds, *self.cfg.birds, *self.cfg.dogs, *self.cfg.cats)
                if n.strip()} - ANIMAL_LABELS

    def other_kinds(self) -> set[str]:
        """Kinds that replace a detector label (squirrel, raccoon...), as opposed to a breed/species of it."""
        return {k.strip().lower() for k in self.cfg.kinds} - ANIMAL_LABELS

    def _text(self, names: list[str]) -> np.ndarray:
        key = tuple(names)
        with self._lock:
            m = self._cache.get(key)
        if m is None:
            rows = []
            for n in names:            # average a few phrasings: steadier than one prompt
                v = self.embedder.texts([t.format(n) for t in TEMPLATES]).mean(axis=0)
                rows.append(v / np.linalg.norm(v))
            m = np.stack(rows)
            with self._lock:
                if len(self._cache) > 16:
                    self._cache.clear()
                self._cache[key] = m
        return m

    def _pick(self, names: list[str], vec: np.ndarray) -> tuple[str, float]:
        logits = self._text(names) @ vec * SCALE
        p = np.exp(logits - logits.max())
        p /= p.sum()
        i = int(p.argmax())
        return names[i], float(p[i])

    def classify(self, label: str, vec: Optional[np.ndarray]) -> Optional[tuple[str, Optional[str], float]]:
        """(kind, subtype or None, certainty) for an animal track's fingerprint, or None to keep the label.

        Every kind and every breed/species competes at once, and a kind's score is the sum of its members
        ("dog" = dog + all the breeds). Comparing kinds against a bare "dog" let fox-like breeds such as a
        Pomeranian come out as "fox".
        """
        if not self.cfg.enabled or vec is None or label not in ANIMAL_LABELS:
            return None
        kinds = [k.strip().lower() for k in self.cfg.kinds if k.strip()]
        if label not in kinds:
            kinds.append(label)
        names, parent = [], []
        for k in kinds:
            for n in [k] + [f.strip().lower() for f in self.fine_list(k) if f.strip()]:
                names.append(n)
                parent.append(k)
        logits = self._text(names) @ vec * SCALE
        p = np.exp(logits - logits.max())
        p /= p.sum()
        by_kind: dict[str, float] = {}
        for k, v in zip(parent, p):
            by_kind[k] = by_kind.get(k, 0.0) + float(v)
        kind = max(by_kind, key=by_kind.get)
        if by_kind[kind] < self.cfg.min_confidence:
            return None
        members = [(float(v), n) for n, k, v in zip(names, parent, p) if k == kind and n != kind]
        sub = None
        if members:
            v, n = max(members)
            if v / by_kind[kind] >= self.cfg.min_confidence:     # sure enough which breed/species, given the kind
                sub = n
        if kind == label and sub is None:
            return None
        return kind, sub, (v / by_kind[kind]) if sub else by_kind[kind]
