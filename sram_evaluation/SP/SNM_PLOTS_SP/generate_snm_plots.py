#!/usr/bin/env python3
"""Generate publication-style SNM plots from measured SRAM bitcell CSV files.

The script expects the directory layout used in this workspace:

    SNM/
      SP00/
      SP01/
      ...
      SP22/

Each measurement needs a pair of files:

    SP SNM<hold|read> sweep Q  [...]
    SP SNM<hold|read> sweep Qn [...]

The sweep-Q file is interpreted as x=Q, y=Qn.  The sweep-Qn file is
interpreted as x=Qn, y=Q and is mapped back to x=Q, y=Qn before computing
SNM.  SNM is computed by rotating both measured curves by 45 degrees and
finding the maximum vertical gap in the two lobes.  The reported SNM is the
smaller lobe gap divided by sqrt(2).
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import math
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SQRT2 = math.sqrt(2.0)
DEFAULT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snm-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="SNM root containing SP00 ... SP22 directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <snm-root>/plots_publication_all.",
    )
    parser.add_argument(
        "--structures",
        nargs="*",
        default=[f"SP{i:02d}" for i in range(23)],
        help="Structures to process, e.g. SP13 SP14. Defaults to SP00..SP22.",
    )
    parser.add_argument(
        "--include-non-ok",
        action="store_true",
        help="Also plot curves where one lobe gap is non-positive.",
    )
    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Do not generate one image per structure/die/mode.",
    )
    parser.add_argument(
        "--skip-overlay",
        action="store_true",
        help="Do not generate structure-level overlay images.",
    )
    return parser.parse_args()


def normalize_structure(name: str) -> str:
    match = re.search(r"sp\s*0*(\d{1,2})", name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse structure from {name!r}")
    return f"SP{int(match.group(1)):02d}"


def structure_from_path(path: Path, root: Path) -> Optional[str]:
    try:
        first = path.relative_to(root).parts[0]
    except ValueError:
        first = path.parts[-2]
    if re.fullmatch(r"SP\d{2}", first, re.IGNORECASE):
        return first.upper()
    match = re.search(r"sp\s*0*(\d{1,2})", path.name, re.IGNORECASE)
    if match:
        return f"SP{int(match.group(1)):02d}"
    return None


def parse_measurement_metadata(path: Path, root: Path) -> Dict[str, object]:
    name = path.name
    structure = structure_from_path(path, root)

    if re.search(r"SNM\s*hold|SNMhold", name, re.IGNORECASE):
        mode = "hold"
    elif re.search(r"SNM\s*read|SNMread", name, re.IGNORECASE):
        mode = "read"
    else:
        mode = None

    if re.search(r"sweep\s+Qn|swp_Qn", name, re.IGNORECASE):
        sweep = "Qn"
    elif re.search(r"sweep\s+Q\s|swp_Q", name, re.IGNORECASE):
        sweep = "Q"
    else:
        sweep = None

    # Later SP06..SP22 naming:
    #   pq57601w8die 4 2 sp13  (8)
    match = re.search(
        r"die\s+(\d+)\s+(\d+)\s+sp\s*0*(\d{1,2})\s+\((\d+)\)",
        name,
        re.IGNORECASE,
    )
    if match:
        die_x, die_y, sp_num, measurement_id = map(int, match.groups())
        return {
            "structure": structure or f"SP{sp_num:02d}",
            "die_x": die_x,
            "die_y": die_y,
            "measurement_id": measurement_id,
            "mode": mode,
            "sweep": sweep,
            "file": str(path),
        }

    # Earlier SP00..SP05 naming:
    #   pq57601w8_sp01 _ 10 2  (7)
    match = re.search(
        r"_sp\s*0*(\d{1,2})\s*_\s*(\d+)\s+(\d+)\s+\((\d+)\)",
        name,
        re.IGNORECASE,
    )
    if match:
        sp_num, die_x, die_y, measurement_id = map(int, match.groups())
        return {
            "structure": structure or f"SP{sp_num:02d}",
            "die_x": die_x,
            "die_y": die_y,
            "measurement_id": measurement_id,
            "mode": mode,
            "sweep": sweep,
            "file": str(path),
        }

    return {
        "structure": structure,
        "die_x": None,
        "die_y": None,
        "measurement_id": None,
        "mode": mode,
        "sweep": sweep,
        "file": str(path),
    }


def read_two_columns(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in text.strip().splitlines() if line.strip()]
    rows: List[Tuple[float, float]] = []
    for line in lines[1:]:
        parts = re.split(r"\t|,", line.strip())
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not rows:
        return np.empty((0, 2), dtype=float)
    return np.array(rows, dtype=float)


def dedupe_sorted_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and abs(float(x[j] - x[i])) < 1e-12:
            j += 1
        xs.append(float(np.mean(x[i:j])))
        ys.append(float(np.mean(y[i:j])))
        i = j
    return np.array(xs), np.array(ys)


def rotate_curve(curve: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q = curve[:, 0]
    qn = curve[:, 1]
    u = (qn - q) / SQRT2
    v = (q + qn) / SQRT2
    order = np.argsort(u)
    return dedupe_sorted_xy(u[order], v[order])


def compute_snm(curve_q: np.ndarray, curve_qn: np.ndarray) -> Dict[str, object]:
    if len(curve_q) < 3 or len(curve_qn) < 3:
        raise ValueError("not enough curve points")

    # sweep Qn is measured as x=Qn, y=Q.  Map it back to x=Q, y=Qn.
    curve_qn_mapped = np.column_stack([curve_qn[:, 1], curve_qn[:, 0]])

    u1, v1 = rotate_curve(curve_q)
    u2, v2 = rotate_curve(curve_qn_mapped)
    lo = max(float(u1.min()), float(u2.min()))
    hi = min(float(u1.max()), float(u2.max()))
    if not lo < hi:
        raise ValueError("no common rotated curve range")

    grid = np.linspace(lo, hi, 25001)
    va = np.interp(grid, u1, v1)
    vb = np.interp(grid, u2, v2)
    diff = vb - va
    left = grid < 0
    right = grid > 0
    if not left.any() or not right.any():
        raise ValueError("missing left or right lobe")

    left_idx = int(np.argmax(diff[left]))
    right_idx = int(np.argmax(-diff[right]))
    left_grid = grid[left]
    right_grid = grid[right]

    left_diag = float(diff[left][left_idx])
    right_diag = float((-diff[right])[right_idx])
    snm_left = left_diag / SQRT2
    snm_right = right_diag / SQRT2
    snm = min(snm_left, snm_right)
    limiting_lobe = "left" if snm_left <= snm_right else "right"
    status = "ok" if left_diag > 0 and right_diag > 0 else "non_positive_lobe_gap"

    if limiting_lobe == "left":
        u_square = float(left_grid[left_idx])
        v_low = float(va[left][left_idx])
        v_high = float(vb[left][left_idx])
    else:
        u_square = float(right_grid[right_idx])
        v_low = float(vb[right][right_idx])
        v_high = float(va[right][right_idx])

    x_low = (v_low - u_square) / SQRT2
    y_low = (v_low + u_square) / SQRT2
    x_high = (v_high - u_square) / SQRT2
    y_high = (v_high + u_square) / SQRT2

    return {
        "snm_v": snm,
        "snm_mv": snm * 1000.0,
        "snm_left_v": snm_left,
        "snm_left_mv": snm_left * 1000.0,
        "snm_right_v": snm_right,
        "snm_right_mv": snm_right * 1000.0,
        "left_diag_v": left_diag,
        "right_diag_v": right_diag,
        "limiting_lobe": limiting_lobe,
        "status": status,
        "curve_qn_mapped": curve_qn_mapped,
        "square": (x_low, y_low, x_high, y_high),
    }


def font_path() -> Optional[Path]:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    return next((path for path in candidates if path.exists()), None)


class PlotStyle:
    scale = 3
    width = 620
    height = 620
    left = 76
    top = 54
    right = 570
    bottom = 540
    xmin = -0.05
    xmax = 1.25
    ymin = -0.05
    ymax = 1.25

    def __init__(self) -> None:
        path = font_path()
        bold = Path(r"C:\Windows\Fonts\arialbd.ttf")

        def load(size: int, use_bold: bool = False) -> ImageFont.ImageFont:
            px = int(size * self.scale)
            if path is None:
                return ImageFont.load_default()
            if use_bold and bold.exists():
                return ImageFont.truetype(str(bold), px)
            return ImageFont.truetype(str(path), px)

        self.title_font = load(17)
        self.label_font = load(20)
        self.tick_font = load(13)
        self.ann_font = load(15)
        self.small_font = load(12)

    def data_x(self, value: float) -> float:
        return self.left + (value - self.xmin) / (self.xmax - self.xmin) * (
            self.right - self.left
        )

    def data_y(self, value: float) -> float:
        return self.bottom - (value - self.ymin) / (self.ymax - self.ymin) * (
            self.bottom - self.top
        )

    def scaled(self, value: float) -> int:
        return int(round(value * self.scale))


def draw_rotated_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    angle: float,
    fill: Tuple[int, int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0] + 18 * PlotStyle.scale
    height = bbox[3] - bbox[1] + 18 * PlotStyle.scale
    layer = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    draw.text(
        (9 * PlotStyle.scale - bbox[0], 9 * PlotStyle.scale - bbox[1]),
        text,
        font=font,
        fill=fill,
    )
    layer = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(
        layer,
        (int(xy[0] - layer.width / 2), int(xy[1] - layer.height / 2)),
    )


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    fill: Tuple[int, int, int],
    width: int,
) -> None:
    draw.line([p0, p1], fill=fill, width=width)
    x0, y0 = p0
    x1, y1 = p1
    angle = math.atan2(y1 - y0, x1 - x0)
    length = 12 * PlotStyle.scale
    spread = math.radians(25)
    for head_angle in (angle + math.pi - spread, angle + math.pi + spread):
        p = (x1 + length * math.cos(head_angle), y1 + length * math.sin(head_angle))
        draw.line([p1, p], fill=fill, width=width)


def create_canvas(style: PlotStyle) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new(
        "RGBA",
        (style.width * style.scale, style.height * style.scale),
        (255, 255, 255, 255),
    )
    return image, ImageDraw.Draw(image, "RGBA")


def finalize_image(image: Image.Image, style: PlotStyle, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    image = image.resize((style.width, style.height), Image.Resampling.LANCZOS).convert("RGB")
    image.save(output, dpi=(300, 300))


def draw_axes(image: Image.Image, draw: ImageDraw.ImageDraw, style: PlotStyle) -> None:
    s = style.scaled
    draw.rectangle(
        [s(style.left), s(style.top), s(style.right), s(style.bottom)],
        outline=(45, 45, 45),
        width=max(1, s(1)),
    )
    for idx in range(7):
        value = round(idx * 0.2, 1)
        x = style.data_x(value)
        y = style.data_y(value)
        label = "0" if abs(value) < 1e-12 else (f"{value:.1f}").rstrip("0").rstrip(".")
        draw.line(
            [(s(x), s(style.bottom)), (s(x), s(style.bottom - 7))],
            fill=(45, 45, 45),
            width=max(1, s(1)),
        )
        draw.line(
            [(s(x), s(style.top)), (s(x), s(style.top + 7))],
            fill=(45, 45, 45),
            width=max(1, s(1)),
        )
        draw.line(
            [(s(style.left), s(y)), (s(style.left + 7), s(y))],
            fill=(45, 45, 45),
            width=max(1, s(1)),
        )
        draw.line(
            [(s(style.right), s(y)), (s(style.right - 7), s(y))],
            fill=(45, 45, 45),
            width=max(1, s(1)),
        )
        draw.text(
            (s(x), s(style.bottom + 15)),
            label,
            font=style.tick_font,
            fill=(55, 55, 55),
            anchor="ma",
        )
        draw.text(
            (s(style.left - 12), s(y)),
            label,
            font=style.tick_font,
            fill=(55, 55, 55),
            anchor="rm",
        )


def plot_curve(
    draw: ImageDraw.ImageDraw,
    style: PlotStyle,
    curve: np.ndarray,
    color: Tuple[int, int, int, int],
    width: float,
) -> None:
    if len(curve) < 2:
        return
    points = [
        (style.scaled(style.data_x(float(x))), style.scaled(style.data_y(float(y))))
        for x, y in curve
    ]
    draw.line(points, fill=color, width=max(1, style.scaled(width)), joint="curve")


def plot_individual(
    row: Dict[str, object],
    curve_q: np.ndarray,
    curve_qn_mapped: np.ndarray,
    square: Tuple[float, float, float, float],
    output: Path,
) -> None:
    style = PlotStyle()
    image, draw = create_canvas(style)
    s = style.scaled

    draw_axes(image, draw, style)
    plot_curve(draw, style, curve_q, (10, 10, 10, 255), 1.35)
    plot_curve(draw, style, curve_qn_mapped, (10, 10, 10, 255), 1.35)

    x0, y0, x1, y1 = square
    xlo, xhi = sorted((x0, x1))
    ylo, yhi = sorted((y0, y1))
    sx0 = style.data_x(xlo)
    sx1 = style.data_x(xhi)
    sy_top = style.data_y(yhi)
    sy_bottom = style.data_y(ylo)
    draw.rectangle(
        [s(sx0), s(sy_top), s(sx1), s(sy_bottom)],
        outline=(0, 0, 0),
        width=max(1, s(1.15)),
    )
    p_bl = (s(sx0), s(sy_bottom))
    p_tr = (s(sx1), s(sy_top))
    draw_arrow(draw, p_bl, p_tr, (0, 0, 0), max(1, s(1.2)))
    draw_arrow(draw, p_tr, p_bl, (0, 0, 0), max(1, s(1.2)))
    draw.text(
        (s((sx0 + sx1) / 2), s((sy_top + sy_bottom) / 2 + 4)),
        "SNM",
        font=style.ann_font,
        fill=(0, 0, 0),
        anchor="mm",
    )

    mode = str(row["mode"])
    draw.text(
        (s(style.data_x(0.57)), s(style.data_y(0.88))),
        "Vo(Vi)",
        font=style.ann_font,
        fill=(0, 0, 0),
        anchor="lm",
    )
    draw.text(
        (s(style.data_x(0.88)), s(style.data_y(0.55 if mode == "hold" else 0.48))),
        "Vi(Vo)",
        font=style.ann_font,
        fill=(0, 0, 0),
        anchor="lm",
    )

    legend_x = 388
    legend_y = 92
    draw.line(
        [(s(legend_x), s(legend_y)), (s(legend_x + 38), s(legend_y))],
        fill=(10, 10, 10),
        width=max(1, s(1.3)),
    )
    draw.text(
        (s(legend_x + 48), s(legend_y)),
        "measured curves",
        font=style.small_font,
        fill=(0, 0, 0),
        anchor="lm",
    )
    draw.rectangle(
        [s(legend_x), s(legend_y + 16), s(legend_x + 38), s(legend_y + 34)],
        outline=(0, 0, 0),
        width=max(1, s(1)),
    )
    draw.text(
        (s(legend_x + 48), s(legend_y + 25)),
        "SNM square",
        font=style.small_font,
        fill=(0, 0, 0),
        anchor="lm",
    )
    draw.text(
        (s(legend_x), s(legend_y + 58)),
        f"SNM = {float(row['snm_mv']):.1f} mV",
        font=style.small_font,
        fill=(0, 0, 0),
        anchor="lm",
    )
    draw.text(
        (s(legend_x), s(legend_y + 78)),
        f"left {float(row['snm_left_mv']):.1f} / right {float(row['snm_right_mv']):.1f} mV",
        font=style.small_font,
        fill=(0, 0, 0),
        anchor="lm",
    )

    title = (
        f"Static noise margin - {row['structure']} {row['mode']} "
        f"die ({row['die_x']},{row['die_y']})"
    )
    draw.text(
        (s((style.left + style.right) / 2), s(24)),
        title,
        font=style.title_font,
        fill=(45, 45, 45),
        anchor="mm",
    )
    draw.text(
        (s((style.left + style.right) / 2), s(590)),
        "Vi (Vo) [V]",
        font=style.label_font,
        fill=(35, 35, 35),
        anchor="mm",
    )
    draw_rotated_text(
        image,
        "Vo (Vi) [V]",
        (s(28), s((style.top + style.bottom) / 2)),
        90,
        (35, 35, 35, 255),
        style.label_font,
    )
    finalize_image(image, style, output)


def color_for_index(index: int, total: int) -> Tuple[int, int, int, int]:
    hue = (index / max(total, 1)) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.55)
    alpha = 70 if total > 20 else 130
    return int(red * 255), int(green * 255), int(blue * 255), alpha


def select_representative_entries(
    entries: List[Dict[str, object]],
    target: int = 9,
) -> List[Dict[str, object]]:
    """Select representative dies across the measured SNM range."""
    if len(entries) <= target:
        return entries

    ranked = sorted(entries, key=lambda item: float(item["snm_mv"]))
    indexes = [
        int(round(index * (len(ranked) - 1) / (target - 1)))
        for index in range(target)
    ]
    selected: List[Dict[str, object]] = []
    seen = set()
    for index in indexes:
        entry = ranked[index]
        key = (entry["die_x"], entry["die_y"], entry["measurement_id"], entry["mode"])
        if key in seen:
            continue
        selected.append(entry)
        seen.add(key)

    if len(selected) < target:
        for entry in ranked:
            key = (entry["die_x"], entry["die_y"], entry["measurement_id"], entry["mode"])
            if key in seen:
                continue
            selected.append(entry)
            seen.add(key)
            if len(selected) == target:
                break

    return selected


def plot_overlay(
    structure: str,
    mode: str,
    entries: List[Dict[str, object]],
    output: Path,
) -> None:
    style = PlotStyle()
    image, draw = create_canvas(style)
    s = style.scaled
    draw_axes(image, draw, style)

    all_entries = sorted(
        entries,
        key=lambda item: (
            int(item["die_x"]),
            int(item["die_y"]),
            int(item["measurement_id"]),
        ),
    )
    selected_from_all = len(all_entries) > 12
    entries = (
        select_representative_entries(all_entries, target=9)
        if selected_from_all
        else all_entries
    )
    for index, entry in enumerate(entries):
        color = color_for_index(index, len(entries))
        plot_curve(draw, style, entry["curve_q"], color, 1.0)
        plot_curve(draw, style, entry["curve_qn_mapped"], color, 1.0)

    draw.text(
        (s((style.left + style.right) / 2), s(24)),
        f"SNM variation - {structure} {mode} {'selected dies' if selected_from_all else 'all dies'}",
        font=style.title_font,
        fill=(45, 45, 45),
        anchor="mm",
    )
    draw.text(
        (s((style.left + style.right) / 2), s(590)),
        "Vi (Vo) [V]",
        font=style.label_font,
        fill=(35, 35, 35),
        anchor="mm",
    )
    draw_rotated_text(
        image,
        "Vo (Vi) [V]",
        (s(28), s((style.top + style.bottom) / 2)),
        90,
        (35, 35, 35, 255),
        style.label_font,
    )

    values = [float(item["snm_mv"]) for item in all_entries]
    legend_x = 386
    legend_y = 86
    if selected_from_all:
        draw.rectangle(
            [s(372), s(68), s(562), s(392)],
            fill=(255, 255, 255, 245),
        )
    draw.text(
        (s(legend_x), s(legend_y)),
        f"{len(entries)} selected dies" if selected_from_all else f"{len(entries)} valid die measurements",
        font=style.small_font,
        fill=(0, 0, 0),
        anchor="lm",
    )
    stats_y = legend_y + 23
    if selected_from_all:
        draw.text(
            (s(legend_x), s(stats_y)),
            f"from {len(all_entries)} valid",
            font=style.small_font,
            fill=(0, 0, 0),
            anchor="lm",
        )
        stats_y += 23
    if values:
        draw.text(
            (s(legend_x), s(stats_y)),
            f"SNM mean {statistics.mean(values):.1f} mV",
            font=style.small_font,
            fill=(0, 0, 0),
            anchor="lm",
        )
        draw.text(
            (s(legend_x), s(stats_y + 23)),
            f"min {min(values):.1f} / max {max(values):.1f} mV",
            font=style.small_font,
            fill=(0, 0, 0),
            anchor="lm",
        )
    curve_legend_y = stats_y + 52
    draw.line(
        [(s(legend_x), s(curve_legend_y)), (s(legend_x + 42), s(curve_legend_y))],
        fill=(60, 60, 60, 180),
        width=max(1, s(1.2)),
    )
    draw.text(
        (s(legend_x + 52), s(curve_legend_y)),
        "Q and Qn mapped curves",
        font=style.small_font,
        fill=(0, 0, 0),
        anchor="lm",
    )

    # The dense structures are reduced to nine representative dies so the
    # legend remains readable.
    if len(entries) <= 12:
        start_y = curve_legend_y + 29
        for index, entry in enumerate(entries[:12]):
            y = start_y + 19 * index
            color = color_for_index(index, len(entries))
            draw.line(
                [(s(legend_x), s(y)), (s(legend_x + 35), s(y))],
                fill=color,
                width=max(1, s(1.4)),
            )
            label = f"die ({entry['die_x']},{entry['die_y']})"
            draw.text(
                (s(legend_x + 45), s(y)),
                label,
                font=style.small_font,
                fill=(0, 0, 0),
                anchor="lm",
            )

    finalize_image(image, style, output)


def collect_measurements(root: Path, structures: Iterable[str]) -> Tuple[List[Tuple[Path, Dict[str, object]]], List[Dict[str, object]]]:
    selected = {normalize_structure(structure) for structure in structures}
    parsed: List[Tuple[Path, Dict[str, object]]] = []
    issues: List[Dict[str, object]] = []
    for structure in sorted(selected):
        directory = root / structure
        if not directory.exists():
            issues.append({"structure": structure, "issue": "missing_structure_directory"})
            continue
        for path in directory.rglob("*.csv"):
            if any(part.lower().startswith("postprocess") for part in path.parts):
                continue
            meta = parse_measurement_metadata(path, root)
            required = (
                meta.get("structure"),
                meta.get("die_x"),
                meta.get("die_y"),
                meta.get("measurement_id"),
                meta.get("mode"),
                meta.get("sweep"),
            )
            if not all(value is not None for value in required):
                issues.append({**meta, "issue": "parse_failed"})
                continue
            parsed.append((path, meta))
    return parsed, issues


def build_entries(
    parsed: List[Tuple[Path, Dict[str, object]]],
    include_non_ok: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    grouped: Dict[Tuple[object, ...], Dict[str, List[Tuple[Path, Dict[str, object]]]]] = {}
    for path, meta in parsed:
        key = (
            meta["structure"],
            meta["die_x"],
            meta["die_y"],
            meta["measurement_id"],
            meta["mode"],
        )
        grouped.setdefault(key, {}).setdefault(str(meta["sweep"]), []).append((path, meta))

    entries: List[Dict[str, object]] = []
    issues: List[Dict[str, object]] = []
    for key, sweeps in sorted(grouped.items()):
        structure, die_x, die_y, measurement_id, mode = key
        q_list = sweeps.get("Q", [])
        qn_list = sweeps.get("Qn", [])
        if len(q_list) != 1 or len(qn_list) != 1:
            issues.append(
                {
                    "structure": structure,
                    "die_x": die_x,
                    "die_y": die_y,
                    "measurement_id": measurement_id,
                    "mode": mode,
                    "issue": "missing_or_duplicate_sweep",
                    "q_count": len(q_list),
                    "qn_count": len(qn_list),
                }
            )
            continue

        q_path, _ = q_list[0]
        qn_path, _ = qn_list[0]
        try:
            curve_q = read_two_columns(q_path)
            curve_qn = read_two_columns(qn_path)
            result = compute_snm(curve_q, curve_qn)
            if result["status"] != "ok" and not include_non_ok:
                issues.append(
                    {
                        "structure": structure,
                        "die_x": die_x,
                        "die_y": die_y,
                        "measurement_id": measurement_id,
                        "mode": mode,
                        "issue": result["status"],
                        "q_file": str(q_path),
                        "qn_file": str(qn_path),
                    }
                )
                continue
            entries.append(
                {
                    "structure": structure,
                    "die_x": die_x,
                    "die_y": die_y,
                    "measurement_id": measurement_id,
                    "mode": mode,
                    "q_file": str(q_path),
                    "qn_file": str(qn_path),
                    "curve_q": curve_q,
                    "curve_qn": curve_qn,
                    **result,
                }
            )
        except Exception as exc:
            issues.append(
                {
                    "structure": structure,
                    "die_x": die_x,
                    "die_y": die_y,
                    "measurement_id": measurement_id,
                    "mode": mode,
                    "issue": f"compute_failed: {exc}",
                    "q_file": str(q_path),
                    "qn_file": str(qn_path),
                }
            )
    return entries, issues


def write_manifest(entries: List[Dict[str, object]], issues: List[Dict[str, object]], output_dir: Path) -> None:
    manifest = output_dir / "plot_manifest.csv"
    fields = [
        "structure",
        "die_x",
        "die_y",
        "measurement_id",
        "mode",
        "snm_mv",
        "snm_left_mv",
        "snm_right_mv",
        "limiting_lobe",
        "status",
        "individual_png",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow({key: entry.get(key, "") for key in fields})

    issue_path = output_dir / "plot_diagnostics.csv"
    issue_fields = [
        "structure",
        "die_x",
        "die_y",
        "measurement_id",
        "mode",
        "issue",
        "q_count",
        "qn_count",
        "q_file",
        "qn_file",
        "file",
    ]
    with issue_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=issue_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(issues)


def main() -> None:
    args = parse_args()
    root = args.snm_root.resolve()
    output_dir = (args.output_dir or (root / "plots_publication_all")).resolve()
    individual_dir = output_dir / "individual"
    overlay_dir = output_dir / "overlay"
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed, parse_issues = collect_measurements(root, args.structures)
    entries, compute_issues = build_entries(parsed, include_non_ok=args.include_non_ok)
    issues = parse_issues + compute_issues

    if not args.skip_individual:
        for entry in entries:
            structure = str(entry["structure"])
            mode = str(entry["mode"])
            output = (
                individual_dir
                / structure
                / mode
                / (
                    f"{structure}_snm_{mode}_die_{entry['die_x']}_{entry['die_y']}"
                    f"_id_{entry['measurement_id']}.png"
                )
            )
            plot_individual(
                entry,
                entry["curve_q"],
                entry["curve_qn_mapped"],
                entry["square"],
                output,
            )
            entry["individual_png"] = str(output)

    if not args.skip_overlay:
        by_structure_mode: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
        for entry in entries:
            key = (str(entry["structure"]), str(entry["mode"]))
            by_structure_mode.setdefault(key, []).append(entry)
        for (structure, mode), group in sorted(by_structure_mode.items()):
            output = overlay_dir / structure / f"{structure}_snm_{mode}_all_dies.png"
            plot_overlay(structure, mode, group, output)

    write_manifest(entries, issues, output_dir)

    print(f"parsed_files={len(parsed)}")
    print(f"plotted_measurements={len(entries)}")
    print(f"diagnostics={len(issues)}")
    print(f"output_dir={output_dir}")
    print(f"individual_dir={individual_dir}")
    print(f"overlay_dir={overlay_dir}")


if __name__ == "__main__":
    main()
