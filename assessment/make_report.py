"""Entry point: generate MD reports from pipeline results.

Usage:
    python -m assessment.make_report
    python -m assessment.make_report --config assessment/configs/report_config.yaml
"""
import argparse

import yaml

from .reporting.md_report import generate_reports


def main(config_path="assessment/configs/report_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    generate_reports(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate assessment reports")
    parser.add_argument("--config", default="assessment/configs/report_config.yaml")
    args = parser.parse_args()
    main(args.config)
