from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def load_single_method(path: Path) -> Dict:
    """
    Load ONE method dict from a JSON file.
    """

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 1:
    if isinstance(data, dict) and "name" in data:
        return data

    # 2:
    if isinstance(data, dict):
        candidate_methods = [
            v for v in data.values()
            if isinstance(v, dict) and "name" in v
        ]
        if len(candidate_methods) == 1:
            return candidate_methods[0]
        elif len(candidate_methods) > 1:
            raise ValueError(
                f"{path} contains multiple methods, "
                f"but the script expects exactly one per JSON file."
            )

    raise ValueError(
        f"Unrecognized JSON format in {path}. "
        f"Expected either a method dict with 'name', "
        f"or a dict with exactly one such method inside."
    )


def prepare_series(
    method: Dict,
    metric_key: str = "acc-flexible",
    eff_key: str = "efficiency",
) -> Tuple[List[str], List[float], List[float]]:
    """
    Convert a method dict into sorted (labels, efficiency, metric) series.

    method example:
    {
        "0.60": {"acc-flexible": ..., "efficiency": ...},
        "0.70": {...},
        "name": "soft token"
    }
    """
    points = []
    for k, v in method.items():
        if k == "name":
            continue
        try:
            x_numeric = float(k)
        except (TypeError, ValueError):
            continue
        points.append((x_numeric, k, v))

    if not points:
        raise ValueError("No valid numeric keys found in method for series preparation.")

    points.sort(key=lambda item: item[0])

    labels = [label for _, label, _ in points]
    efficiency = [info[eff_key] for _, _, info in points]
    metric = [info[metric_key] for _, _, info in points]
    return labels, efficiency, metric


def setup_matplotlib_style() -> None:
    """Set a clean, paper-like style."""
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update(
        {
            "figure.figsize": (6.0, 4.0),
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _thousands_formatter(x, pos):
    # e.g. transform 56938 into "57k"
    if x >= 1000:
        return f"{int(x/1000)}k"
    return str(int(x))


def plot_acc_efficiency(
    json_paths: Sequence[Path],
    output_path: Path,
    metric_key: str = "acc-flexible",
    title: str | None = None,
) -> None:
    """
    Plot accuracy vs efficiency curves for multiple methods.
    """

    setup_matplotlib_style()
    fig, ax = plt.subplots()

    # color / marker cycle
    color_cycle = plt.cm.tab10.colors  # 10 color max
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "8"]

    method_names: List[str] = []

    for idx, json_path in enumerate(json_paths):
        method = load_single_method(json_path)
        method_name = method.get("name", json_path.stem)
        method_names.append(method_name)

        labels, efficiency, metric = prepare_series(
            method, metric_key=metric_key
        )

        color = color_cycle[idx % len(color_cycle)]
        marker = markers[idx % len(markers)]

        ax.plot(
            efficiency,
            metric,
            marker=marker,
            linestyle="-",
            linewidth=1.6,
            markersize=5,
            color=color,
            label=method_name,
        )

        # note keys above the data points（e.g. "0.60", "0.70"）
        if len(metric) > 1:
            y_range = max(metric) - min(metric)
            y_offset = y_range * 0.015
        else:
            y_offset = 0.002

        for x, y, point_label in zip(efficiency, metric, labels):
            ax.text(
                x,
                y + y_offset,
                point_label,
                fontsize=8,
                ha="center",
                va="bottom",
                color=color,
            )

    ax.set_xlabel("Num Total NFE")
    metric_pretty = metric_key.replace("acc-", "").capitalize()  # flexible -> Flexible
    ax.set_ylabel(f"Accuracy ({metric_pretty})")

    ax.xaxis.set_major_formatter(FuncFormatter(_thousands_formatter))

    if title is None:
        # joined_names = ", ".join(method_names)
        joined_names = "Llada-8B-Base"
        ax.set_title(f"Accuracy vs Efficiency Comparison\n({joined_names})")
    else:
        ax.set_title(title)

    ax.legend(frameon=True, loc="best")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(json_files: Sequence[str]) -> None:
    if not json_files:
        print("Usage: python plot_acc_efficiency.py method1.json [method2.json ...]")
        sys.exit(1)

    json_paths = [Path(p) for p in json_files]

    # Edit output name
    output_path = Path.cwd() / "acc_efficiency_comparison.png"
    # Edit plotting metric
    metric_key = "acc-flexible"

    plot_acc_efficiency(json_paths, output_path, metric_key=metric_key)

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
