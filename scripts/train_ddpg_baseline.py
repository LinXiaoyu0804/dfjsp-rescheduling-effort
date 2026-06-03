from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

from _bootstrap import REPO_ROOT  # noqa: F401

from src.baselines.ddpg import (
    DEFAULT_RULE_CANDIDATES,
    DDPGRulePolicy,
    build_rule_priority_matrix,
    extract_ddpg_state_features,
)
from src.baselines.dispatching import dispatching_release_count
from src.train.checkpoint import save_checkpoint
from src.utils.config import load_merged_config
from src.utils.seed import set_global_seed


@dataclass(slots=True)
class DDPGTrainingSample:
    features: torch.Tensor
    rule_scores: torch.Tensor
    release_labels: torch.Tensor


class DDPGTrainingDataset(Dataset[DDPGTrainingSample]):
    def __init__(self, samples: list[DDPGTrainingSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DDPGTrainingSample:
        return self.samples[index]


def collate_training_samples(batch: list[DDPGTrainingSample]) -> list[DDPGTrainingSample]:
    return batch


def build_training_dataset(dataset_path: str, rule_candidates: list[str]) -> DDPGTrainingDataset:
    raw_samples = torch.load(dataset_path, map_location="cpu", weights_only=False)
    samples: list[DDPGTrainingSample] = []
    for sample in raw_samples:
        graph = sample["graph_tensors"]
        features = extract_ddpg_state_features(graph)
        op_ids = list(graph["op_ids"])
        rule_scores = build_rule_priority_matrix(
            instance=graph["instance"],
            snapshot=graph["snapshot"],
            op_ids=op_ids,
            rule_candidates=rule_candidates,
        )
        release_labels = sample["release_labels"].float()
        if rule_scores.shape[0] != release_labels.numel():
            raise ValueError(
                f"Mismatched training sample lengths: rule_scores={rule_scores.shape[0]} "
                f"release_labels={release_labels.numel()}"
            )
        samples.append(
            DDPGTrainingSample(
                features=features,
                rule_scores=rule_scores,
                release_labels=release_labels,
            )
        )
    return DDPGTrainingDataset(samples)


def _predict_release_scores(
    model: nn.Module,
    sample: DDPGTrainingSample,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = model(sample.features.unsqueeze(0).to(device)).squeeze(0)
    rule_scores = sample.rule_scores.to(device)
    if rule_scores.numel() == 0:
        release_scores = torch.empty((0,), dtype=torch.float32, device=device)
        candidate_mask = torch.zeros((0,), dtype=torch.bool, device=device)
    else:
        release_scores = torch.matmul(rule_scores, weights).clamp(min=1e-4, max=1.0 - 1e-4)
        candidate_mask = rule_scores.sum(dim=1) > 0
    return release_scores, weights, candidate_mask


def _release_f1(
    release_scores: torch.Tensor,
    release_labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    release_fraction: float,
) -> float:
    labels = release_labels.detach().cpu()
    candidate_idx = candidate_mask.detach().cpu().nonzero(as_tuple=False).reshape(-1)
    if candidate_idx.numel() == 0:
        return 1.0 if float(labels.sum().item()) == 0.0 else 0.0

    candidate_scores = release_scores.detach().cpu()[candidate_idx]
    release_count = dispatching_release_count(int(candidate_idx.numel()))
    release_count = min(
        int(candidate_idx.numel()),
        max(release_count, int(round(int(candidate_idx.numel()) * release_fraction))),
    )
    if release_count <= 0:
        predicted_release = set()
    else:
        topk_local = torch.topk(candidate_scores, k=release_count).indices
        predicted_release = {int(candidate_idx[idx].item()) for idx in topk_local}
    teacher_release = {int(idx) for idx, label in enumerate(labels.tolist()) if float(label) > 0.5}

    if not predicted_release and not teacher_release:
        return 1.0
    tp = len(predicted_release & teacher_release)
    fp = len(predicted_release - teacher_release)
    fn = len(teacher_release - predicted_release)
    denom = (2 * tp + fp + fn)
    return 0.0 if denom == 0 else (2 * tp) / denom


def evaluate(model: nn.Module, loader: DataLoader, device: str, release_fraction: float, size_penalty: float) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_f1 = 0.0
    total = 0
    with torch.no_grad():
        for batch in loader:
            for sample in batch:
                release_scores, _, candidate_mask = _predict_release_scores(model, sample, device)
                labels = sample.release_labels.to(device)
                if release_scores.numel() == 0:
                    loss = torch.tensor(0.0, device=device)
                else:
                    bce = nn.functional.binary_cross_entropy(release_scores, labels)
                    size_alignment = torch.abs(release_scores.sum() - labels.sum()) / max(1, labels.numel())
                    loss = bce + size_penalty * size_alignment
                total_loss += float(loss.item())
                total_f1 += _release_f1(release_scores, labels, candidate_mask, release_fraction)
                total += 1
    return total_loss / max(1, total), total_f1 / max(1, total)


def train_ddpg_model(cfg: dict) -> Path:
    ddpg_cfg = cfg["ddpg_baseline"]
    rule_candidates = list(ddpg_cfg.get("rule_candidates", DEFAULT_RULE_CANDIDATES))
    seed = int(cfg["experiment"]["seed"])
    set_global_seed(seed)

    dataset = build_training_dataset(ddpg_cfg["dataset_path"], rule_candidates)
    if len(dataset) <= 1:
        train_set = dataset
        val_set = dataset
    else:
        val_size = max(1, int(len(dataset) * float(ddpg_cfg.get("val_ratio", 0.2))))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

    batch_size = int(ddpg_cfg.get("batch_size", 16))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_training_samples)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_training_samples)

    device = str(cfg["experiment"].get("device", "cpu"))
    release_fraction = float(ddpg_cfg.get("release_fraction", 1.0 / 3.0))
    size_penalty = float(ddpg_cfg.get("size_penalty", 0.1))
    model = DDPGRulePolicy(
        input_dim=int(ddpg_cfg.get("input_dim", 65)),
        hidden_dim=int(ddpg_cfg.get("hidden_dim", 64)),
        dropout=float(ddpg_cfg.get("dropout", 0.1)),
        num_rules=len(rule_candidates),
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(ddpg_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(ddpg_cfg.get("weight_decay", 1e-4)),
    )

    best_val = float("inf")
    best_val_f1 = 0.0
    patience = 0
    checkpoint_path = Path(ddpg_cfg["checkpoint_path"])

    for _epoch in range(int(ddpg_cfg.get("num_epochs", 160))):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            valid_items = 0
            for sample in batch:
                release_scores, _, _ = _predict_release_scores(model, sample, device)
                labels = sample.release_labels.to(device)
                if release_scores.numel() == 0:
                    continue
                bce = nn.functional.binary_cross_entropy(release_scores, labels)
                size_alignment = torch.abs(release_scores.sum() - labels.sum()) / max(1, labels.numel())
                batch_loss = batch_loss + bce + size_penalty * size_alignment
                valid_items += 1
            if valid_items == 0:
                continue
            batch_loss.backward()
            optimizer.step()

        val_loss, val_f1 = evaluate(model, val_loader, device, release_fraction, size_penalty)
        if val_loss < best_val:
            best_val = val_loss
            best_val_f1 = val_f1
            patience = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer=optimizer,
                extra={
                    "rule_candidates": rule_candidates,
                    "input_dim": int(ddpg_cfg.get("input_dim", 65)),
                    "release_fraction": release_fraction,
                    "best_val_loss": best_val,
                    "best_val_f1": best_val_f1,
                    "seed": seed,
                },
            )
        else:
            patience += 1
            if patience >= int(ddpg_cfg.get("early_stopping_patience", 20)):
                break

    print(f"Saved DDPG baseline checkpoint to {checkpoint_path}")
    print(f"Rule candidates: {rule_candidates}")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Best validation release F1: {best_val_f1:.4f}")
    return checkpoint_path


def train_ddpg_if_needed(config_paths: list[str], force: bool = False) -> Path:
    cfg = load_merged_config(*config_paths)
    checkpoint_path = Path(cfg["ddpg_baseline"]["checkpoint_path"])
    if checkpoint_path.exists() and not force:
        print(f"[ddpg-train] Reusing existing checkpoint: {checkpoint_path}")
        return checkpoint_path
    return train_ddpg_model(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Gui et al. style DDPG baseline with weighted dispatching-rule aggregation."
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/default.yaml", "configs/baselines/ddpg.yaml"],
    )
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    train_ddpg_if_needed(list(args.config), force=bool(args.force_retrain))


if __name__ == "__main__":
    main()
