from pathlib import Path
import re
import shutil

from transformers.utils import logging

logger = logging.get_logger(__name__)


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


def sorted_checkpoints(output_dir, use_mtime=False):
    checkpoints = []
    for checkpoint_dir in Path(output_dir).glob("checkpoint-*"):
        if not checkpoint_dir.is_dir():
            continue
        if use_mtime:
            ordering_key = checkpoint_dir.stat().st_mtime
        else:
            match = CHECKPOINT_PATTERN.search(checkpoint_dir.name)
            ordering_key = (
                int(match.group(1)) if match else checkpoint_dir.stat().st_mtime
            )
        checkpoints.append((ordering_key, str(checkpoint_dir)))
    return [checkpoint for _, checkpoint in sorted(checkpoints)]


def delete_checkpoint(checkpoint_dir):
    """
    Deletes a specified checkpoint directory.

    Args:
        checkpoint_dir: The path to the checkpoint directory to delete.
    """
    try:
        logger.info(f"Deleting checkpoint {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)
        logger.info("Deletion successful.")
    except FileNotFoundError:
        logger.warning(f"Checkpoint {checkpoint_dir} does not exist.")


def cleanup_checkpoints(trainer, training_args):
    """
    Cleans up checkpoints based on training arguments.

    Args:
        trainer: The Hugging Face Trainer instance.
        training_args: TrainingArguments used for training configuration.
    """
    logger.info("Cleaning up checkpoints!")
    output_dir = training_args.output_dir
    checkpoints_sorted = sorted_checkpoints(output_dir, use_mtime=False)
    logger.info(f"Current checkpoints: {checkpoints_sorted}")

    if training_args.keep_checkpoints in {"none", "eval"}:
        # Delete individual checkpoints but preserve evaluation results
        # Assumes checkpoints and evaluation results are stored separately
        for checkpoint in checkpoints_sorted:
            delete_checkpoint(checkpoint)

    logger.info("Successful completion.")
