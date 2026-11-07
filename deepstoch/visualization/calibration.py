"""Dependency-light SVG calibration visualizations."""

from __future__ import annotations

from html import escape
from pathlib import Path


def _scale(value: float, minimum: float, maximum: float, start: float, length: float) -> float:
    if maximum <= minimum:
        return start + length / 2
    return start + (value - minimum) / (maximum - minimum) * length


def render_calibration_svg(
    fold_scores: list[dict[str, object]],
    output_path: str | Path,
    *,
    title: str = "DeepStoch rolling calibration",
) -> Path:
    """Write a self-contained SVG with coverage and interval-width panels."""
    if not fold_scores:
        raise ValueError("at least one fold score is required")
    required = {"empirical_coverage", "nominal_coverage", "mean_interval_width"}
    if any(not required.issubset(score) for score in fold_scores):
        raise ValueError("fold scores are missing calibration fields")
    width, height = 1000, 620
    left, right, top, panel_height = 85, 35, 80, 190
    plot_width = width - left - right
    coverages = [float(score["empirical_coverage"]) for score in fold_scores]
    nominals = [float(score["nominal_coverage"]) for score in fold_scores]
    interval_widths = [float(score["mean_interval_width"]) for score in fold_scores]
    coverage_min = min(0.0, min(coverages), min(nominals))
    coverage_max = max(1.0, max(coverages), max(nominals))
    width_min, width_max = min(interval_widths), max(interval_widths)
    if width_max <= width_min:
        width_max = width_min + 1.0

    def points(values: list[float], minimum: float, maximum: float, y_top: float) -> str:
        return " ".join(
            f"{_scale(index, 0, max(1, len(values) - 1), left, plot_width):.1f},{_scale(value, minimum, maximum, y_top + panel_height, -panel_height):.1f}"
            for index, value in enumerate(values)
        )

    nominal = nominals[0]
    nominal_y = _scale(nominal, coverage_min, coverage_max, top + panel_height, -panel_height)
    fold_labels = "".join(
        f'<text x="{_scale(index, 0, max(1, len(fold_scores) - 1), left, plot_width):.1f}" y="{top + panel_height + 24}" text-anchor="middle" font-size="11">{index + 1}</text>'
        for index in range(len(fold_scores))
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#0b1220"/>
<text x="{left}" y="38" fill="#f8fafc" font-family="Arial, sans-serif" font-size="24" font-weight="bold">{escape(title)}</text>
<text x="{left}" y="61" fill="#94a3b8" font-family="Arial, sans-serif" font-size="13">Chronological fold diagnostics; coverage target is nominal, width is predictive interval width</text>
<rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="#111c31" stroke="#334155"/>
<line x1="{left}" y1="{nominal_y:.1f}" x2="{width-right}" y2="{nominal_y:.1f}" stroke="#f59e0b" stroke-dasharray="6 5"/>
<polyline points="{points(coverages, coverage_min, coverage_max, top)}" fill="none" stroke="#38bdf8" stroke-width="3"/>
{fold_labels}
<text x="{left}" y="{top - 10}" fill="#e2e8f0" font-family="Arial, sans-serif" font-size="15">Empirical coverage</text>
<text x="{width-right-5}" y="{nominal_y-7:.1f}" text-anchor="end" fill="#fbbf24" font-family="Arial, sans-serif" font-size="12">nominal {nominal:.0%}</text>
<text x="{left}" y="{top + panel_height + 43}" fill="#94a3b8" font-family="Arial, sans-serif" font-size="12">Rolling fold</text>
<rect x="{left}" y="{top + panel_height + 85}" width="{plot_width}" height="{panel_height}" fill="#111c31" stroke="#334155"/>
<polyline points="{points(interval_widths, width_min, width_max, top + panel_height + 85)}" fill="none" stroke="#a78bfa" stroke-width="3"/>
<text x="{left}" y="{top + panel_height + 75}" fill="#e2e8f0" font-family="Arial, sans-serif" font-size="15">Mean interval width</text>
<text x="{left}" y="{height - 20}" fill="#64748b" font-family="Arial, sans-serif" font-size="12">DeepStoch calibration report</text>
</svg>'''
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8")
    return destination
