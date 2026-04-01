#!/usr/bin/env python3
"""Generate 18,000 training examples across 4 categories for Oracle fine-tuning."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oracle.training.data_generator import CATEGORIES, TrainingDataGenerator  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "training"


async def main() -> None:
    generator = TrainingDataGenerator(model="claude-3-5-sonnet-20241022")

    for category, config in CATEGORIES.items():
        count = config["count"]
        print(f"\n{'='*60}")
        print(f"Generating {count} examples for: {category}")
        print(f"{'='*60}")

        examples = await generator.generate_batch(category=category, count=count, batch_size=10)
        print(f"  Generated {len(examples)} examples")

    # Save full dataset
    output_path = OUTPUT_DIR / "oracle_training_18k.jsonl"
    generator.save_dataset(output_path)

    stats = generator.stats()
    print(f"\n{'='*60}")
    print(f"Dataset complete: {stats['total']} total examples")
    for cat, cnt in stats["by_category"].items():
        print(f"  {cat}: {cnt}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
