"""Recognise delivery services (Amazon, UPS, FedEx, USPS) on vehicles and people, from the CLIP fingerprint each
track already has for visual search: no extra image work.

The picture is compared with descriptions of each service's vehicles (brown UPS package car, blue Amazon van,
FedEx purple/orange, USPS eagle mail truck) or uniforms, together with many descriptions of ordinary vehicles and
people. A service is named only when its descriptions clearly win over the ordinary ones.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

log = logging.getLogger("theeye.delivery")

VEHICLE_LABELS = {"car", "truck", "bus"}
PERSON_LABELS = {"person"}
TEMPLATES = ["{}", "a photo of {}.", "a security camera photo of {}."]
SCALE = 100.0
MIN_SIDE = 0.07          # box width and height, as a share of the frame
EMERGENCY = {"Police", "Fire", "EMT"}
EMERGENCY_MIN = 0.98     # a wrong police/fire/ambulance tag is worse than a missed one: higher bar

SERVICES: dict[str, dict[str, list[str]]] = {
    "Amazon": {
        "vehicle": ["an Amazon Prime delivery van with the Amazon smile logo", "a blue Amazon delivery van",
                    "a white Amazon delivery van with a blue stripe and the Amazon logo"],
        # The vest is what shows at doorstep distance (blue, reflective silver stripes). These two took a real
        # driver at night from 42% to 100% without tagging any of ~1,750 other people. A generic "person carrying
        # a box" description was tried and dropped: it tagged people taking boxes out to the bins.
        "person": ["an Amazon delivery driver wearing a blue and yellow safety vest carrying a package",
                   "a delivery driver in an Amazon vest holding a parcel",
                   "a delivery driver wearing a blue safety vest with reflective silver stripes",
                   "a person in a blue hi-vis vest carrying a cardboard box to the front door at night"],
    },
    "UPS": {
        "vehicle": ["a brown UPS delivery truck", "a brown UPS package car with the gold UPS shield logo"],
        "person": ["a UPS delivery driver in a brown uniform", "a UPS driver in brown shirt and brown shorts carrying a box"],
    },
    "FedEx": {
        "vehicle": ["a white FedEx delivery truck with the purple and orange FedEx logo",
                    "a FedEx Ground delivery van with purple and green FedEx logo"],
        "person": ["a FedEx courier in a purple and black uniform", "a FedEx delivery driver carrying a package"],
    },
    "USPS": {
        "vehicle": ["a white USPS mail truck with the blue eagle logo", "a United States Postal Service mail delivery vehicle",
                    "a white postal delivery truck with red and blue stripes"],
        "person": ["a USPS mail carrier in a blue postal uniform carrying mail",
                   "a mailman with a mail bag in a blue-grey postal uniform"],
    },
    # Emergency services: same machinery, same 0.95 bar.
    "Police": {
        "vehicle": ["a police car with a light bar on the roof", "a black and white police patrol car",
                    "a police SUV with emergency lights", "a police cruiser with POLICE written on the side"],
        "person": ["a police officer in uniform with a duty belt and badge", "a uniformed cop wearing a police vest"],
    },
    "Fire": {
        "vehicle": ["a red fire engine", "a red fire truck with a ladder", "a fire department truck with emergency lights"],
        "person": ["a firefighter in turnout gear and helmet", "a firefighter in a tan and yellow fire suit"],
    },
    "EMT": {
        "vehicle": ["an ambulance", "a white ambulance with red stripes and emergency lights",
                    "a paramedic ambulance with the star of life symbol"],
        "person": ["a paramedic in uniform carrying a medical bag", "an EMT pushing a stretcher"],
    },
}
ORDINARY: dict[str, list[str]] = {
    "vehicle": ["a regular passenger car", "a sedan", "an SUV", "a pickup truck", "a minivan", "a white work van",
                "a plain white cargo van", "a box truck", "a school bus", "a city bus", "a parked car on a residential street",
                "a car driving down the street", "a garbage truck", "a tow truck", "a red car", "a black car",
                "a silver car", "a white car", "a blue car", "a brown car", "a blue sedan", "a blue hatchback",
                "a dark brown sedan", "a dark blue SUV", "a white SUV", "a white pickup truck", "a delivery-sized white truck without logos",
                "a red pickup truck", "a red SUV", "a black SUV", "a white van with a ladder rack", "a dark sedan",
                "a car's red tail lights at dusk", "a blurry red light at night", "a car with its headlights on at night",
                "a dark car and a white car parked at dusk"],
    "person": ["a person walking", "a person in casual clothes", "a man in a t-shirt", "a woman in a jacket", "a child",
               "a person walking a dog", "a jogger", "a person in a hoodie", "a person in a brown jacket",
               "a person in a blue shirt", "a person carrying a bag", "a neighbor in the yard", "a person on a bicycle",
               "a person in a dark jacket", "a person in a high-visibility work vest", "a person in a navy blue shirt"],
}


class DeliveryClassifier:
    def __init__(self, embedder, min_confidence: float = 0.95):
        self.embedder = embedder
        self.min_confidence = min_confidence
        self._cache: dict[str, tuple[np.ndarray, list[Optional[str]]]] = {}
        self._lock = threading.Lock()

    def _text(self, kind: str) -> tuple[np.ndarray, list[Optional[str]]]:
        with self._lock:
            hit = self._cache.get(kind)
        if hit:
            return hit
        prompts, owner = [], []
        for name, d in SERVICES.items():
            for p in d[kind]:
                prompts.append(p)
                owner.append(name)
        for p in ORDINARY[kind]:
            prompts.append(p)
            owner.append(None)
        rows = []
        for p in prompts:          # average a few phrasings of each description
            v = self.embedder.texts([t.format(p) for t in TEMPLATES]).mean(axis=0)
            rows.append(v / np.linalg.norm(v))
        hit = (np.stack(rows), owner)
        with self._lock:
            self._cache[kind] = hit
        return hit

    def scores(self, label: str, vec: np.ndarray) -> Optional[dict[Optional[str], float]]:
        kind = "vehicle" if label in VEHICLE_LABELS else "person" if label in PERSON_LABELS else None
        if kind is None or vec is None:
            return None
        m, owner = self._text(kind)
        logits = m @ vec * SCALE
        p = np.exp(logits - logits.max())
        p /= p.sum()
        out: dict[Optional[str], float] = {}
        for o, v in zip(owner, p):
            out[o] = out.get(o, 0.0) + float(v)
        return out

    def classify(self, label: str, vec: Optional[np.ndarray], bbox: Optional[list[float]] = None
                 ) -> Optional[tuple[str, float]]:
        """(service, certainty) when one service's descriptions clearly win, else None. Boxes too small to show
        markings or a uniform are skipped: tiny far-off shapes matched services almost at random."""
        if bbox and len(bbox) == 4 and (bbox[2] - bbox[0] < MIN_SIDE or bbox[3] - bbox[1] < MIN_SIDE):
            return None
        s = self.scores(label, vec) if vec is not None else None
        if not s:
            return None
        best = max((k for k in s if k), key=s.get)
        need = max(self.min_confidence, EMERGENCY_MIN) if best in EMERGENCY else self.min_confidence
        return (best, s[best]) if s[best] >= need else None
