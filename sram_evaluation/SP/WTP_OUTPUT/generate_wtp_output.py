#!/usr/bin/env python3
"""Compute WTP output CSV from SP-sorted measurement files."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_ROOT = Path(__file__).resolve().parent
THRESHOLD_V = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wtp-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="WTP root containing SP00 ... SP22 directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV. Defaults to <wtp-root>/WTP_OUTPUT_CSV.csv.",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="Diagnostics CSV. Defaults to <wtp-root>/WTP_OUTPUT_DIAGNOSTICS.csv.",
    )
    return parser.parse_args()


def parse_file_metadata(path: Path) -> Optional[Dict[str, object]]:
    name = path.name
    step_match = re.search(r"step_(5&6|9&10)", name, re.IGNORECASE)
    die_match = re.search(r"_SP\s*_\s*(\d+)\s+(\d+)", name, re.IGNORECASE)
    subsite_match = re.search(r"_Subsite_(\d+)", name, re.IGNORECASE)
    measurement_match = re.search(r"\((\d+)\)", name)
    if not (step_match and die_match and subsite_match):
        return None

    subsite = int(subsite_match.group(1))
    structure = f"SP{subsite:02d}"
    return {
        "structure": structure,
        "sp_index": subsite,
        "die_x": int(die_match.group(1)),
        "die_y": int(die_match.group(2)),
        "measurement_id": int(measurement_match.group(1)) if measurement_match else "",
        "step": step_match.group(1),
        "file": str(path),
        "file_name": name,
    }


def read_curve(path: Path) -> Tuple[str, List[Tuple[float, float]]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return "", []

    header = lines[0].strip()
    rows: List[Tuple[float, float]] = []
    for line in lines[1:]:
        parts = re.split(r"\t|,", line.strip())
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return header, rows


def crossing_voltage(
    rows: List[Tuple[float, float]],
    threshold: float = THRESHOLD_V,
) -> Tuple[Optional[float], str, float, float]:
    if not rows:
        return None, "empty_curve", math.nan, math.nan

    y_values = [row[1] for row in rows]
    y_min = min(y_values)
    y_max = max(y_values)

    for x, y in rows:
        if abs(y - threshold) < 1e-15:
            return x, "valid_crossing", y_min, y_max

    for (x0, y0), (x1, y1) in zip(rows, rows[1:]):
        d0 = y0 - threshold
        d1 = y1 - threshold
        if d0 == d1:
            continue
        if d0 * d1 < 0:
            frac = (threshold - y0) / (y1 - y0)
            return x0 + frac * (x1 - x0), "valid_crossing", y_min, y_max

    if y_min > threshold:
        return None, "already_high", y_min, y_max
    if y_max < threshold:
        return None, "no_flip_low", y_min, y_max
    return None, "no_crossing_found", y_min, y_max


def collect_step_results(root: Path) -> Tuple[Dict[Tuple[object, ...], Dict[str, Dict[str, object]]], List[Dict[str, object]]]:
    grouped: Dict[Tuple[object, ...], Dict[str, Dict[str, object]]] = {}
    diagnostics: List[Dict[str, object]] = []

    for directory in sorted(root.glob("SP[0-9][0-9]")):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.csv")):
            meta = parse_file_metadata(path)
            if meta is None:
                diagnostics.append(
                    {
                        "structure": directory.name,
                        "issue": "parse_failed",
                        "file": str(path),
                    }
                )
                continue
            if str(meta["structure"]) != directory.name:
                diagnostics.append(
                    {
                        **meta,
                        "issue": "subsite_folder_mismatch",
                        "folder": directory.name,
                    }
                )

            header, rows = read_curve(path)
            value, status, y_min, y_max = crossing_voltage(rows)
            step_result = {
                **meta,
                "wtp_v": value,
                "status": status,
                "header": header.replace("\t", "|"),
                "y_min": y_min,
                "y_max": y_max,
                "points": len(rows),
            }
            key = (
                meta["structure"],
                meta["sp_index"],
                meta["die_x"],
                meta["die_y"],
                meta["measurement_id"],
            )
            grouped.setdefault(key, {})[str(meta["step"])] = step_result
            if status != "valid_crossing":
                diagnostics.append({**step_result, "issue": status})

    return grouped, diagnostics


def fmt_number(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def build_output_rows(grouped: Dict[Tuple[object, ...], Dict[str, Dict[str, object]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, steps in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2], item[0][3], item[0][4])):
        structure, sp_index, die_x, die_y, measurement_id = key
        step_56 = steps.get("5&6")
        step_910 = steps.get("9&10")
        value_56 = step_56.get("wtp_v") if step_56 else None
        value_910 = step_910.get("wtp_v") if step_910 else None
        valid_values = [
            value
            for value in (value_56, value_910)
            if isinstance(value, float) and not math.isnan(value)
        ]
        worst_min = min(valid_values) if valid_values else None
        avg = sum(valid_values) / len(valid_values) if valid_values else None

        if len(valid_values) == 2:
            point_status = "valid_both"
        elif len(valid_values) == 1:
            point_status = "valid_one_direction"
        else:
            point_status = "invalid_both"

        def mv(value: object) -> object:
            return value * 1000.0 if isinstance(value, float) and not math.isnan(value) else ""

        rows.append(
            {
                "structure": structure,
                "sp_index": sp_index,
                "die_x": die_x,
                "die_y": die_y,
                "die": f"({die_x},{die_y})",
                "measurement_id": measurement_id,
                "wtp_step_5_6_v": fmt_number(value_56),
                "wtp_step_5_6_mv": mv(value_56),
                "wtp_step_9_10_v": fmt_number(value_910),
                "wtp_step_9_10_mv": mv(value_910),
                "wtp_worst_min_v": fmt_number(worst_min),
                "wtp_worst_min_mv": mv(worst_min),
                "wtp_avg_v": fmt_number(avg),
                "wtp_avg_mv": mv(avg),
                "step_5_6_status": step_56["status"] if step_56 else "missing",
                "step_9_10_status": step_910["status"] if step_910 else "missing",
                "point_status": point_status,
                "step_5_6_file": step_56["file_name"] if step_56 else "",
                "step_9_10_file": step_910["file_name"] if step_910 else "",
                "step_5_6_path": step_56["file"] if step_56 else "",
                "step_9_10_path": step_910["file"] if step_910 else "",
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    root = args.wtp_root.resolve()
    output = (args.output or (root / "WTP_OUTPUT_CSV.csv")).resolve()
    diagnostics = (args.diagnostics or (root / "WTP_OUTPUT_DIAGNOSTICS.csv")).resolve()

    grouped, diagnostic_rows = collect_step_results(root)
    output_rows = build_output_rows(grouped)

    output_fields = [
        "structure",
        "sp_index",
        "die_x",
        "die_y",
        "die",
        "measurement_id",
        "wtp_step_5_6_v",
        "wtp_step_5_6_mv",
        "wtp_step_9_10_v",
        "wtp_step_9_10_mv",
        "wtp_worst_min_v",
        "wtp_worst_min_mv",
        "wtp_avg_v",
        "wtp_avg_mv",
        "step_5_6_status",
        "step_9_10_status",
        "point_status",
        "step_5_6_file",
        "step_9_10_file",
        "step_5_6_path",
        "step_9_10_path",
    ]
    diagnostic_fields = [
        "structure",
        "sp_index",
        "die_x",
        "die_y",
        "measurement_id",
        "step",
        "issue",
        "status",
        "wtp_v",
        "y_min",
        "y_max",
        "points",
        "header",
        "folder",
        "file_name",
        "file",
    ]
    write_csv(output, output_rows, output_fields)
    write_csv(diagnostics, diagnostic_rows, diagnostic_fields)

    valid_both = sum(1 for row in output_rows if row["point_status"] == "valid_both")
    valid_one = sum(1 for row in output_rows if row["point_status"] == "valid_one_direction")
    invalid = sum(1 for row in output_rows if row["point_status"] == "invalid_both")
    print(f"rows={len(output_rows)}")
    print(f"valid_both={valid_both}")
    print(f"valid_one_direction={valid_one}")
    print(f"invalid_both={invalid}")
    print(f"diagnostics={len(diagnostic_rows)}")
    print(f"output={output}")
    print(f"diagnostics_csv={diagnostics}")


if __name__ == "__main__":
    main()
