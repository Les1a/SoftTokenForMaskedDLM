from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

NFE_PATTERN = re.compile(r"Total NFE(?: is|:)\s*([\d,\.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate gsm8k evaluation metrics (strict-match, flexible-extract, NFE) "
            "from a directory into a JSON file."
        )
    )
    parser.add_argument(
        "input_dir",
        default="/home/hice1/kxia39/workspace/Llada_dual_branch/eval_results_soft_token/parallel_dual",
        type=Path,
        help="Directory that contains gsm8k_XX subfolders (e.g. eval_results_soft_token/base_parallel_dual).",
    )
    parser.add_argument(
        "output",
        default="prob1.5max_soft_token.json",
        type=Path,
        help="Destination JSON file. Existing content is updated in-place for the provided method key.",
    )
    parser.add_argument(
        "--task",
        default="gsm8k",
        help="Target task prefix to scan (e.g.: gsm8k).",
    )

    # not needed, modify manually is ok
    parser.add_argument(
        "--method-key",
        default="method1",
        help="Key to use in the JSON output (default: method1).",
    )
    # not needed, modify manually is ok
    parser.add_argument(
        "--method-name",
        default=None,
        help="Human readable name for the method. Defaults to the method key if omitted.",
    )
    return parser.parse_args()


def read_summary(summary_path: Path) -> int:
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        match = NFE_PATTERN.search(line)
        if match:
            return int(float(match.group(1).replace(",", "")))
    raise ValueError(f"Failed to find 'Total NFE' in {summary_path}")


def find_results_file(task_dir: Path) -> Path:
    candidates = sorted(task_dir.rglob("results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No results_*.json found under {task_dir}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def read_metrics(results_path: Path, task: str) -> Dict[str, float]:
    with results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    try:
        metrics = payload["results"][task]
    except KeyError as exc:
        raise KeyError(f"Task '{task}' not found in {results_path}") from exc
    return {
        "acc-flexible": float(metrics["exact_match,flexible-extract"]),
        "acc-strict": float(metrics["exact_match,strict-match"]),
    }


def collect_task_entries(input_dir: Path, task: str) -> Dict[str, dict]:
    prefix = f"{task}_"
    entries = {}
    for subdir in sorted(
        (path for path in input_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)),
        key=lambda path: float(path.name.split("_", 1)[1]),
    ):
        threshold = subdir.name.split("_", 1)[1]
        summary_path = subdir / "summary.txt"
        if not summary_path.exists():
            raise FileNotFoundError(f"{summary_path} is missing")
        nfe = read_summary(summary_path)
        results_file = find_results_file(subdir)
        metrics = read_metrics(results_file, task)
        entries[threshold] = {
            "acc-flexible": metrics["acc-flexible"],
            "acc-strict": metrics["acc-strict"],
            "efficiency": nfe,
        }
    if not entries:
        raise ValueError(f"No '{prefix}*' folders found in {input_dir}")
    return entries


def load_existing(output_path: Path) -> Dict[str, dict]:
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_output(output_path: Path, content: Dict[str, dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(content, f, indent=4, sort_keys=True)


def main() -> None:
    args = parse_args()
    method_payload = {"name": args.method_name or args.method_key}
    method_payload.update(collect_task_entries(args.input_dir, args.task))

    data = load_existing(args.output)
    data[args.method_key] = method_payload
    write_output(args.output, data)


if __name__ == "__main__":
    main()
