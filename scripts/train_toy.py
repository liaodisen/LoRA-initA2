#!/usr/bin/env python3
"""Reference MNIST reconstruction of the toy experiments in Sections 4.1/C.2.

The original toy source was not archived. See docs/EXPERIMENTS.md for the
explicit choices needed to make the stated zero-backbone model trainable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ToyLoRA(nn.Module):
    def __init__(self, width=256, rank=8, beta=1.0, init="AA", input_dim=784):
        super().__init__()
        self.width = width
        self.register_buffer("win", torch.randn(width, input_dim))
        self.register_buffer("wout", torch.randn(10, width) / width)
        scale = 1.0 if init == "AA" else 0.5
        self.a = nn.Parameter(torch.randn(rank, width) * beta / width**scale)
        self.b = nn.Parameter(torch.zeros(width, rank))
        if init == "AB_reset":
            nn.init.normal_(self.b, std=beta / width**0.5)
        self.register_buffer("a0", self.a.detach().clone())
        self.register_buffer("b0", self.b.detach().clone())
        self.reset = init == "AB_reset"

    def forward(self, x, return_features=False):
        z = torch.relu(x @ self.win.T)
        za = z @ self.a.T
        zb = za @ self.b.T
        adapted = zb
        if self.reset:
            adapted = adapted - (z @ self.a0.T) @ self.b0.T
        # clamp_min is ReLU with subgradient 1 at exactly zero. A zero
        # backbone and B0=0 otherwise give zero gradients with torch.relu.
        logits = adapted.clamp_min(0) @ self.wout.T
        return (logits, za, zb) if return_features else logits


def load_mnist(max_samples=None):
    from datasets import load_dataset

    data = load_dataset("ylecun/mnist")
    result = []
    for split in ["train", "test"]:
        rows = data[split]
        if max_samples is not None:
            rows = rows.select(range(min(max_samples, len(rows))))
        x = np.stack(
            [
                np.asarray(row["image"], dtype=np.float32).reshape(-1) / 255
                for row in rows
            ]
        )
        y = np.asarray(rows["label"], dtype=np.int64)
        result.append(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--init", choices=["A", "AA", "AB_reset"], default="AA")
    parser.add_argument("--lr_a", type=float, default=2**-7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic fake images only for smoke checks",
    )
    args = parser.parse_args()
    if min(args.width, args.rank, args.epochs, args.batch_size) <= 0 or args.lr_a <= 0:
        parser.error(
            "Dimensions, epochs, batch size, and learning rate must be positive"
        )
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    if args.synthetic:
        train = TensorDataset(
            torch.rand(32, 784, generator=generator), torch.arange(32) % 10
        )
        validation = train
    else:
        train, validation = load_mnist(args.max_samples)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ToyLoRA(args.width, args.rank, args.beta, args.init).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": [model.a], "lr": args.lr_a},
            {"params": [model.b], "lr": args.width * args.lr_a},
        ],
        betas=(0.0, 0.0),
        eps=1e-8,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "output_dir": str(args.output_dir),
        "reconstructed": True,
        "relu_subgradient_at_zero": 1,
        "image_scale": "[0,1]",
        "adam_betas": [0, 0],
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    probe = next(iter(validation_loader))[0].to(device)
    history, features = [], []
    step = 0

    def record_features():
        with torch.no_grad():
            _, za, zb = model(probe, return_features=True)
        features.append(
            {
                "step": step,
                "za_norm_squared_over_width": za.square().sum(-1).mean().item()
                / args.width,
                "zb_norm_squared_over_width": zb.square().sum(-1).mean().item()
                / args.width,
            }
        )

    record_features()
    for epoch in range(args.epochs):
        losses = []
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite toy loss at step {step}")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
            if step <= 100:
                record_features()
            if args.max_steps > 0 and step >= args.max_steps:
                break
        validation_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in validation_loader:
                logits = model(x.to(device))
                validation_loss += nn.functional.cross_entropy(
                    logits, y.to(device), reduction="sum"
                ).item()
                correct += (logits.argmax(-1).cpu() == y).sum().item()
                total += len(y)
        history.append(
            {
                "epoch": epoch + 1,
                "step": step,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss / total,
                "accuracy": correct / total,
            }
        )
        print(history[-1], flush=True)
        if args.max_steps > 0 and step >= args.max_steps:
            break
    (args.output_dir / "toy_history.json").write_text(json.dumps(history, indent=2))
    (args.output_dir / "toy_features.json").write_text(json.dumps(features, indent=2))
    metrics = {
        **history[-1],
        "best_validation_loss": min(row["validation_loss"] for row in history),
        "reconstructed": True,
        "synthetic": args.synthetic,
    }
    (args.output_dir / "toy_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
