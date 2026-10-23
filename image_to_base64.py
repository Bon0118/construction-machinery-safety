"""Convert an image file to a base64 string.

Usage:
    python image_to_base64.py alarm_outputs/example.jpg
    python image_to_base64.py alarm_outputs/example.jpg --output image_base64.txt
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path


def image_to_base64(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an image file to base64.")
    parser.add_argument("image", help="Image file path.")
    parser.add_argument("--output", "-o", help="Optional output text file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoded = image_to_base64(Path(args.image))
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
