#!/usr/bin/env python3
"""Generate and save the Oracle metrics report to reports/metrics_report.md."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oracle.reports.metrics_report import MetricsReportGenerator


async def main() -> None:
    output_dir = Path(__file__).resolve().parent.parent / "reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "metrics_report.md"

    generator = MetricsReportGenerator()
    await generator.initialize()
    markdown = await generator.generate_markdown()

    output_path.write_text(markdown, encoding="utf-8")
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
