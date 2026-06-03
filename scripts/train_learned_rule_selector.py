from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from _bootstrap import REPO_ROOT  # noqa: F401

from src.baselines.dispatching import dispatching_release_decision
from src.baselines.learned_rule_selector import (
    DEFAULT_RULE_CANDIDATES,
    LearnedRuleSelector,
    extract_rule_selector_features,
)
from src.train.checkpoint import save_checkpoint
from src.utils.config import load_merged_config
from src.utils.seed import set_global_seed


def _release_f1(pred_release: set[int], teacher_release: set[int]) -> float:
    if not pred_release and not teacher_release:
        return 1.0
    tp = len(pred_release & teacher_release)
    fp = len(pred_release - teacher_release)
    fn = len(teacher_release - pred_release)
    denom = (2 * tp + fp + fn)
    return 0.0 if denom == 0 else (2 * tp) / denom


def _best_rule_label(sample: dict, rule_candidates: list[str]) -> int:
    graph = sample["graph_tensors"]
    op_ids = list(graph["op_ids"])
    teacher_release = {
        op_id
        for op_id, label in zip(op_ids, sample["release_labels"].tolist())
        if float(label) > 0.5
    }
    instance = graph["instance"]
    incumbent = graph["incumbent"]
    snapshot = graph["snapshot"]

    best_rule_idx = 0
    best_score = -1.0
    best_release_count = None
    for idx, rule in enumerate(rule_candidates):
        baseline = dispatching_release_decision(instance, incumbent, snapshot, rule=rule)
        pred_release = set(baseline.decision.released_op_ids)
        score = _release_f1(pred_release, teacher_release)
        release_count = len(pred_release)
        if (
            score > best_score
            or (score == best_score and (best_release_count is None or release_count < best_release_count))
        ):
            best_score = score
            best_release_count = release_count
            best_rule_idx = idx
    return best_rule_idx


def build_training_tensors(dataset_path: str, rule_candidates: list[str]) -> tuple[torch.Tensor, torch.Tensor, Counter]:
    raw_samples = torch.load(dataset_path, map_location="cpu", weights_only=False)
    xs = []
    ys = []
    counter: Counter = Counter()
    for sample in raw_samples:
        xs.append(extract_rule_selector_features(sample["graph_tensors"]))
        label = _best_rule_label(sample, rule_candidates)
        ys.append(label)
        counter[rule_candidates[label]] += 1
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long), counter


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == y).sum().item())
            total += int(x.size(0))
    return total_loss / max(1, total), correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight learned rule-selector baseline.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/default.yaml", "configs/baselines/learned_rule_selector.yaml"],
    )
    args = parser.parse_args()

    cfg = load_merged_config(*args.config)
    selector_cfg = cfg["rule_selector_baseline"]
    rule_candidates = list(selector_cfg.get("rule_candidates", DEFAULT_RULE_CANDIDATES))
    set_global_seed(int(cfg["experiment"]["seed"]))

    x, y, label_counter = build_training_tensors(selector_cfg["dataset_path"], rule_candidates)
    dataset = TensorDataset(x, y)

    if len(dataset) <= 1:
        train_set = dataset
        val_set = dataset
    else:
        val_size = max(1, int(len(dataset) * float(selector_cfg.get("val_ratio", 0.2))))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=int(selector_cfg.get("batch_size", 16)), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=int(selector_cfg.get("batch_size", 16)), shuffle=False)

    device = str(cfg["experiment"].get("device", "cpu"))
    model = LearnedRuleSelector(
        input_dim=int(selector_cfg.get("input_dim", x.shape[1])),
        hidden_dim=int(selector_cfg.get("hidden_dim", 64)),
        dropout=float(selector_cfg.get("dropout", 0.1)),
        num_rules=len(rule_candidates),
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(selector_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(selector_cfg.get("weight_decay", 1e-4)),
    )
    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    patience = 0
    best_val_acc = 0.0
    checkpoint_path = Path(selector_cfg["checkpoint_path"])

    for epoch in range(int(selector_cfg.get("num_epochs", 160))):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            best_val_acc = val_acc
            patience = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer=optimizer,
                extra={
                    "rule_candidates": rule_candidates,
                    "input_dim": int(selector_cfg.get("input_dim", x.shape[1])),
                    "label_distribution": dict(label_counter),
                    "best_val_loss": best_val,
                    "best_val_acc": best_val_acc,
                },
            )
        else:
            patience += 1
            if patience >= int(selector_cfg.get("early_stopping_patience", 20)):
                break

    print(f"Saved learned rule selector checkpoint to {checkpoint_path}")
    print(f"Rule candidates: {rule_candidates}")
    print(f"Label distribution: {dict(label_counter)}")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
