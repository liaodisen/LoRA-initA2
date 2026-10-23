"""Backward-compatible imports for the src-layout trainer module."""

from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from trainer import (  # noqa: E402
    LoraPlusTrainer,
    LoraPlusTrainingArguments,
    create_loraplus_optimizer,
    get_module,
)

__all__ = [
    "LoraPlusTrainer",
    "LoraPlusTrainingArguments",
    "create_loraplus_optimizer",
    "get_module",
]
