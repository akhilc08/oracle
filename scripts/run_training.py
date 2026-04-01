#!/usr/bin/env python3
"""Upload dataset to Modal volume and trigger LoRA fine-tuning."""

import subprocess
import sys
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "training" / "oracle_training_18k.jsonl"
MODAL_VOLUME = "oracle-dataset"
TRAINER_PATH = Path(__file__).resolve().parent.parent / "src" / "oracle" / "training" / "modal_trainer.py"


def upload_dataset() -> None:
    """Upload the training dataset to Modal volume."""
    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Run scripts/generate_training_data.py first.")
        sys.exit(1)

    print(f"Uploading {DATASET_PATH} to Modal volume '{MODAL_VOLUME}'...")
    subprocess.run(
        [
            "modal", "volume", "put", MODAL_VOLUME,
            str(DATASET_PATH), "training_data.jsonl",
        ],
        check=True,
    )
    print("Upload complete.")


def run_training() -> None:
    """Trigger Modal training job."""
    print(f"Starting Modal training: {TRAINER_PATH}")
    subprocess.run(
        ["modal", "run", str(TRAINER_PATH)],
        check=True,
    )
    print("Training complete.")


def main() -> None:
    upload_dataset()
    run_training()


if __name__ == "__main__":
    main()
