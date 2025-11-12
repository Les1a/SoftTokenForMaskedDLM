from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_acc_efficiency(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_series(method: dict) -> Tuple[List[str], List[float], List[float], List[float]]:
    points = [(float(k), k, v) for k, v in method.items() if k != "name"]
    points.sort(key=lambda item: item[0])
    labels = [label for _, label, _ in points]
    efficiency = [info["efficiency"] for _, _, info in points]
    acc_flexible = [info["acc-flexible"] for _, _, info in points]
    acc_strict = [info["acc-strict"] for _, _, info in points]
    return labels, efficiency, acc_flexible, acc_strict


def plot_acc_efficiency(data: Dict[str, dict], output_path: Path) -> None:
    colors = {"method1": "#1f77b4", "method2": "#d62728"}
    plt.figure(figsize=(8, 5))

    for key in ("method1", "method2"):
        method = data[key]
        labels, efficiency, acc_flexible, acc_strict = prepare_series(method)
        label_base = method["name"]

        plt.plot(
            efficiency,
            acc_flexible,
            marker="o",
            color=colors[key],
            label=f"{label_base} flexible",
        )
        # plt.plot(
        #     efficiency,
        #     acc_strict,
        #     marker="s",
        #     linestyle="--",
        #     color=colors[key],
        #     label=f"{label_base} strict",
        # )
        for eff, acc, point_label in zip(efficiency, acc_flexible, labels):
            plt.text(
                eff,
                acc + 0.002,
                point_label,
                fontsize=8,
                color=colors[key],
                ha="center",
            )
        # for eff, acc, point_label in zip(efficiency, acc_strict, labels):
        #     plt.text(
        #         eff,
        #         acc + 0.005,
        #         point_label,
        #         fontsize=8,
        #         color=colors[key],
        #         ha="center",
        #     )

    plt.xlabel("Time cost")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Efficiency Comparison")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def main(root_name) -> None:
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / f"{root_name}.json"
    output_path = script_dir / f"{root_name}.png"
    data = load_acc_efficiency(data_path)
    plot_acc_efficiency(data, output_path)


if __name__ == "__main__":
    main("gsm8k_llada_inst_baseline_vs_soft")
