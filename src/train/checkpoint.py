from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizer=None, extra: dict | None = None) -> None:
    payload = {"model_state": model.state_dict(), "extra": extra or {}}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None) -> dict:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload.get("extra", {})
