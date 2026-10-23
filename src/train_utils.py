from transformers.utils import logging

logger = logging.get_logger(__name__)


def train_model(trainer, training_args, data_args, train_dataset, last_checkpoint=None):
    """
    Manages the training process, handling resumption from checkpoints and errors.

    Args:
        trainer: The Hugging Face Trainer instance.
        training_args: TrainingArguments for training configuration.
        data_args: DataTrainingArguments for data configuration.
        train_dataset: The dataset used for training.
    """
    checkpoint = (
        training_args.resume_from_checkpoint or last_checkpoint
    )  # Simplified conditional assignment
    # Failed runs must exit with their original exception, rather than silently
    # switching to another checkpoint or restarting an experiment.
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics
    max_train_samples = data_args.max_train_samples or len(
        train_dataset
    )  # Use or for default value
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.save_model()  # Save the model and tokenizer
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
