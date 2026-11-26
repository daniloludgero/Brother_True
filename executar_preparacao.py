"""Executa a preparação com suporte a campos CSV muito grandes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

csv.field_size_limit(2**31 - 1)

from preparar_dados import convert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dados_preparados"))
    args = parser.parse_args()
    print(json.dumps(convert(args.inputs, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
