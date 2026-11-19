"""CLIP image/text embeddings for visual ("person in a red jacket") search."""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

log = logging.getLogger("theeye.clip")


class Embedder:
    def __init__(self, model: str, pretrained: str, device: str):
        self.model_name, self.pretrained, self.device = model, pretrained, device
        self._model = self._preprocess = self._tok = None
        self._lock = threading.Lock()
        self.dim = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch
        dev = self.device if torch.cuda.is_available() else "cpu"
        m, _, pre = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained,
                                                          device=dev)
        m.eval()
        self._model, self._preprocess, self._tok = m, pre, open_clip.get_tokenizer(self.model_name)
        self.device = dev
        log.info("CLIP %s loaded on %s", self.model_name, dev)

    def set_device(self, device: str) -> None:
        """Move the model between GPU and CPU while running (frees its GPU memory when moved to the CPU)."""
        import torch
        with self._lock:
            dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
            if self._model is not None and dev != self.device:
                try:
                    self._model.to(dev)
                except Exception as e:  # noqa: BLE001
                    # e.g. the GPU is full (another program's model): a move that fails part way leaves some
                    # layers on each device, and every picture after that failed. Put it all back on the CPU.
                    log.warning("CLIP: couldn't move to %s (%s); staying on the CPU", dev, str(e).splitlines()[0][:120])
                    self._model.to("cpu")
                    dev = "cpu"
                if dev == "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.device = dev
        log.info("CLIP on %s", self.device)

    def image(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        import torch
        from PIL import Image
        with self._lock:
            self._load()
            img = Image.fromarray(bgr[:, :, ::-1])
            x = self._preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                v = self._model.encode_image(x).float()
            v = torch.nn.functional.normalize(v, dim=-1)[0].cpu().numpy().astype(np.float32)
            self.dim = v.shape[0]
            return v

    def texts(self, items: list[str]) -> np.ndarray:
        """Normalized embeddings for several exact texts (no prompt added), one row each."""
        import torch
        with self._lock:
            self._load()
            t = self._tok(items).to(self.device)
            with torch.no_grad():
                v = self._model.encode_text(t).float()
            return torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float32)

    def text(self, s: str) -> np.ndarray:
        import torch
        with self._lock:
            self._load()
            t = self._tok([f"a photo of {s}"]).to(self.device)
            with torch.no_grad():
                v = self._model.encode_text(t).float()
            return torch.nn.functional.normalize(v, dim=-1)[0].cpu().numpy().astype(np.float32)
