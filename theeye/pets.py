"""Named pets: dogs and cats recognised by how they look. Same flow as faces (theeye.people); the shared part is
theeye.looks."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .looks import NamedLooks, tight_look

PET_LABELS = ("dog", "cat")      # detector labels a pet can come out as (a fluffy dog is sometimes a "cat")
# not a sighting since named a skunk, raccoon... (by hand or after an example): never offered as a pet
NOT_WILD = (" AND detection_id NOT IN (SELECT id FROM detections WHERE kind IS NOT NULL"
            " AND kind NOT IN ('dog','cat'))")


class Pets(NamedLooks):
    REG, LOOKS, OWNER, PREFIX = "pets", "pet_looks", "pet_id", "pet"
    LABELS = PET_LABELS
    MATCH, MARGIN, SUGGEST, MIN_PX = 0.82, 0.03, 0.70, 24
    FILTER = NOT_WILD

    add_pet = NamedLooks.add
    remove_pet = NamedLooks.remove


def look_of(hub, det_id: int) -> Optional[np.ndarray]:
    """The animal's picture fingerprint: the stored one when the Pets scan already made it, else made now."""
    r = hub.db.one("SELECT embedding FROM pet_looks WHERE detection_id=?", (det_id,))
    if r:
        return np.frombuffer(r["embedding"], dtype=np.float32)
    look = tight_look(hub, det_id, Pets.MIN_PX)
    return look[0] if look else None
