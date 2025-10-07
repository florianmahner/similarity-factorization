from argparse import ArgumentParser
from pathlib import Path


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "slurm_test.txt"
    marker.write_text("ok\n")
    print("ok")


if __name__ == "__main__":
    main()

