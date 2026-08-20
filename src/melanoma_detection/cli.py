"""Command-line interface for the complete pipeline."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import (
    DEFAULT_CSV_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_IMPORTANCE_PATH,
    DEFAULT_MODEL_PATH,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="melanoma-detection",
        description="Segment skin lesions and run an experimental classifier.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    masks = commands.add_parser("segment", help="Create and cache lesion masks.")
    masks.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    features = commands.add_parser(
        "build-dataset", help="Create masks and a CSV feature table."
    )
    features.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    features.add_argument("--output", type=Path, default=DEFAULT_CSV_PATH)

    train = commands.add_parser("train", help="Train and evaluate the classifier.")
    train.add_argument("--dataset", type=Path, default=DEFAULT_CSV_PATH)
    train.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    train.add_argument(
        "--importance-output", type=Path, default=DEFAULT_IMPORTANCE_PATH
    )

    predict = commands.add_parser("predict", help="Classify one image.")
    predict.add_argument("image", type=Path)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    predict.add_argument(
        "--show", action="store_true", help="Open a window with the lesion boundary."
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(arguments)
    try:
        if args.command == "segment":
            from .dataset import build_dataset

            build_dataset(data_dir=args.data_dir, masks_only=True)
        elif args.command == "build-dataset":
            from .dataset import build_dataset

            build_dataset(data_dir=args.data_dir, csv_path=args.output)
        elif args.command == "train":
            from .model import train_model

            train_model(
                csv_path=args.dataset,
                model_path=args.output,
                importance_path=args.importance_output,
            )
        elif args.command == "predict":
            from .prediction import show_prediction

            show_prediction(args.image, args.model, args.show)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
