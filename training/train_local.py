import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch_geometric.loader import DataLoader

from stream_dataset import SakkirinaStreamDataset
from ValueNetwork import TributeValueNetwork
from StateParser import NODE_DIM, GLOBAL_DIM

# Global feature index 13 is the prestige clock (see StateParser.extract_global_context).
PRESTIGE_CLOCK_GLOBAL_INDEX = 13
PRESTIGE_BUCKETS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, float("inf"))]


def worker_init_fn(worker_id):
    """Workers implmented with AI to speed up training"""
    import random
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def bucket_index(prestige_clock):
    for i, (lo, hi) in enumerate(PRESTIGE_BUCKETS):
        if lo <= prestige_clock < hi:
            return i
    return len(PRESTIGE_BUCKETS) - 1  # prestige_clock == inf edge case, shouldn't happen


def bucket_label(i):
    lo, hi = PRESTIGE_BUCKETS[i]
    return f"[{lo:.2f}, inf)" if hi == float("inf") else f"[{lo:.2f}, {hi:.2f})"


def print_bucketed_metrics(probs, targets, prestige_clocks):
    """probs/targets/prestige_clocks are flat per-sample lists covering one
    full validation pass. Prints loss, accuracy, ROC AUC, and Brier score per
    prestige-clock bucket, each alongside that bucket's majority-class
    baseline accuracy for context."""
    buckets = defaultdict(lambda: {"probs": [], "targets": []})
    for p, t, pc in zip(probs, targets, prestige_clocks):
        b = buckets[bucket_index(pc)]
        b["probs"].append(p)
        b["targets"].append(t)

    print(f"  {'bucket':<14}{'n':>7}{'loss':>9}{'acc':>9}{'auc':>9}{'brier':>9}{'baseline':>10}")
    eps = 1e-7
    for i in range(len(PRESTIGE_BUCKETS)):
        data = buckets.get(i)
        if not data or len(data["targets"]) == 0:
            print(f"  {bucket_label(i):<14}{0:>7}{'--':>9}{'--':>9}{'--':>9}{'--':>9}{'--':>10}")
            continue
        t = np.array(data["targets"], dtype=np.float64)
        p = np.array(data["probs"], dtype=np.float64)
        p_clipped = np.clip(p, eps, 1 - eps)
        bce = float(-(t * np.log(p_clipped) + (1 - t) * np.log(1 - p_clipped)).mean())
        acc = float(((p >= 0.5).astype(np.float64) == t).mean())
        majority_class = 1.0 if t.mean() >= 0.5 else 0.0
        baseline = float((t == majority_class).mean())
        brier = float(brier_score_loss(t, p))
        auc = roc_auc_score(t, p) if len(np.unique(t)) > 1 else float("nan")
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "n/a (1 class)"
        print(f"  {bucket_label(i):<14}{len(t):>7}{bce:>9.4f}{acc*100:>8.2f}%{auc_str:>10}{brier:>9.4f}{baseline*100:>9.2f}%")


def run_validation(model, val_loader, criterion, device):
    model.eval()
    val_losses, val_accs = [], []
    all_probs, all_targets, all_prestige_clocks = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch)
            target = batch.y.view(-1, 1).float()
            loss = criterion(logits, target)

            probs = torch.sigmoid(logits)
            predictions = (probs >= 0.5).float()
            batch_accuracy = (predictions == target).float().mean().item()

            val_losses.append(loss.item())
            val_accs.append(batch_accuracy)

            all_probs.extend(probs.view(-1).cpu().tolist())
            all_targets.extend(target.view(-1).cpu().tolist())
            all_prestige_clocks.extend(batch.u[:, PRESTIGE_CLOCK_GLOBAL_INDEX].cpu().tolist())

    e_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
    e_val_acc = float(np.mean(val_accs)) if val_accs else 0.0
    return e_val_loss, e_val_acc, all_probs, all_targets, all_prestige_clocks


def train_model(train_dir, val_dir, epochs, batch_size, lr, num_workers, out_dir):
    print("Starting training run")
    os.makedirs(out_dir, exist_ok=True)

    device = pick_device()
    print(f"Using device: {device}")

    model = TributeValueNetwork(node_in_dim=NODE_DIM, global_in_dim=GLOBAL_DIM)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    model.to(device)

    train_dataset = SakkirinaStreamDataset(train_dir)
    val_dataset = SakkirinaStreamDataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                               worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=min(num_workers, 2),
                             worker_init_fn=worker_init_fn)

    smoothing_window = 50
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    best_model_path = os.path.join(out_dir, "best_model.pth")

    for epoch in range(epochs):
        model.train()
        train_losses, train_accs = [], []
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("- training")

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            logits = model(batch)
            target = batch.y.view(-1, 1).float()
            loss = criterion(logits, target)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            batch_accuracy = (preds == target).float().mean().item()

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(batch_accuracy)

            if (batch_idx + 1) % smoothing_window == 0:
                window_loss = float(np.mean(train_losses[-smoothing_window:])) if train_losses[-smoothing_window:] else 0.0
                window_acc = float(np.mean(train_accs[-smoothing_window:])) if train_accs[-smoothing_window:] else 0.0
                print(f"  batch {batch_idx + 1:04d}: {window_loss:.4f} loss, {window_acc * 100:.2f}% acc")

        print("- validation")
        e_val_loss, e_val_acc, val_probs, val_targets, val_prestige_clocks = run_validation(
            model, val_loader, criterion, device)

        scheduler.step(e_val_loss)

        epoch_time = (time.time() - start_time) / 60
        e_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        e_train_acc = float(np.mean(train_accs)) if train_accs else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(e_train_loss)
        history["train_acc"].append(e_train_acc)
        history["val_loss"].append(e_val_loss)
        history["val_acc"].append(e_val_acc)
        history["lr"].append(current_lr)

        print(f"  finished in {epoch_time:.1f} min  (lr={current_lr:.2e})")
        print(f"  train loss {e_train_loss:.4f} | acc {e_train_acc * 100:.2f}%")
        print(f"  val   loss {e_val_loss:.4f} | acc {e_val_acc * 100:.2f}%")
        print(f"  val metrics by prestige clock (global[{PRESTIGE_CLOCK_GLOBAL_INDEX}]):")
        print_bucketed_metrics(val_probs, val_targets, val_prestige_clocks)

        torch.save(model.state_dict(), os.path.join(out_dir, f"deepsets_value_network_epoch{epoch+1}.pth"))

        if e_val_loss < best_val_loss:
            best_val_loss = e_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  new best val loss {best_val_loss:.4f} -- saved {best_model_path}")

        with open(os.path.join(out_dir, "training_metrics.json"), "w") as f:
            json.dump(history, f)

    print(f"\nDone. Best val loss: {best_val_loss:.4f} ({best_model_path})")
    return best_model_path


def main():
    parser = argparse.ArgumentParser(description="Train the Sakkirina value network from a game-aware train/val split.")
    parser.add_argument("--train-dir", required=True, help="Directory of train shards (tools/split_dataset.py's train/)")
    parser.add_argument("--val-dir", required=True, help="Directory of val shards (tools/split_dataset.py's val/)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", default=".", help="Where to write checkpoints and training_metrics.json")
    args = parser.parse_args()

    train_model(args.train_dir, args.val_dir, args.epochs, args.batch_size, args.lr, args.num_workers, args.out_dir)


if __name__ == "__main__":
    main()
