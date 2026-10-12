#!/usr/bin/env python3
"""Compare polars-cubecl benchmark results against cuDF reference numbers.

Usage:
    # Run polars-cubecl benchmarks and generate JSON:
    cargo bench -p cudf-bench -- --save-baseline polars-cubecl

    # (Optional) Run cudf nvbench binaries and capture output to a text file.
    # Then:
    python3 compare.py --cudf-results cudf_nvbench_output.txt \
                       --cubecl-dir target/criterion

Expected cudf nvbench output format (one line per benchmark):
    bench_name  <N>  <median_ms>  <mean_ms>

The script normalizes names and produces a markdown table comparing
median latencies for each operation at each size.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_cudf_file(path: str) -> dict[str, float]:
    """Parse cudf nvbench text output into {bench_name: median_ms}."""
    results: dict[str, float] = {}
    pattern = re.compile(
        r"^\s*(\S+)\s+(\d[\d_]*)\s+([\d.]+)\s+([\d.]+)"
    )
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            match = pattern.match(line)
            if match:
                name = match.group(1)
                size = int(match.group(2).replace("_", ""))
                median_ms = float(match.group(3))
                key = f"{name}_{size}"
                results[key] = median_ms
    return results


def parse_criterion_dir(criterion_dir: Path) -> dict[str, dict[str, float]]:
    """Parse criterion's JSON output into nested dict.

    Structure: {group: {benchmark_id: median_ms}}.
    """
    results: dict[str, dict[str, float]] = {}
    if not criterion_dir.exists():
        print(f"warning: criterion dir not found: {criterion_dir}", file=sys.stderr)
        return results

    for group_dir in criterion_dir.iterdir():
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name
        group_results: dict[str, float] = {}
        for bench_dir in group_dir.iterdir():
            if not bench_dir.is_dir():
                continue
            estimates = bench_dir / "new" / "estimates.json"
            if not estimates.exists():
                estimates = bench_dir / "base" / "estimates.json"
            if not estimates.exists():
                continue
            with open(estimates, encoding="utf-8") as fh:
                data = json.load(fh)
            median_ns = data.get("median", {}).get("point_estimate", 0.0)
            median_ms = median_ns / 1_000_000.0
            group_results[bench_dir.name] = median_ms
        if group_results:
            results[group_name] = group_results
    return results


def format_table(
    cubecl: dict[str, dict[str, float]],
    cudf: dict[str, float] | None,
) -> str:
    """Format a markdown comparison table."""
    lines: list[str] = []
    lines.append("# polars-cubecl vs cuDF Performance Comparison")
    lines.append("")
    lines.append("| Operation | Rows | CubeCL (ms) | Polars CPU (ms) | cuDF (ms) | CubeCL/CPU |")
    lines.append("|---|---|---|---|---|---|")

    for group_name, benches in sorted(cubecl.items()):
        group_label = group_name.replace("_", " ").title()
        for bench_id, median_ms in sorted(benches.items()):
            # Extract row count from bench_id.
            size_match = re.search(r"(\d+)\s*$", bench_id)
            rows = int(size_match.group(1)) if size_match else 0
            rows_str = f"{rows:,}" if rows else "?"

            # Check if this is the CPU baseline.
            is_cpu = "polars_cpu" in bench_id or "cpu" in bench_id.split("_")[0:2]
            if is_cpu:
                cpu_ms = median_ms
                # Find matching cubecl entry.
                cubecl_id = bench_id.replace("polars_cpu", "cubecl")
                cubecl_ms = benches.get(cubecl_id, 0.0)
                if cubecl_ms == 0.0:
                    # Some benchmarks don't have a direct pairing; skip.
                    continue
            else:
                # This is the cubecl entry; extract its timing and find CPU.
                cubecl_ms = median_ms
                cpu_id = bench_id.replace("cubecl", "polars_cpu")
                cpu_ms = benches.get(cpu_id, 0.0)
                if cpu_ms == 0.0:
                    continue

            ratio = cubecl_ms / cpu_ms if cpu_ms > 0 else 0.0
            ratio_str = f"{ratio:.2f}x"

            # cuDF reference.
            cudf_ms = 0.0
            if cudf:
                for key, val in cudf.items():
                    if group_label.replace(" ", "_").lower() in key.lower():
                        cudf_ms = val
                        break
            cudf_str = f"{cudf_ms:.2f}" if cudf_ms > 0 else "—"

            bench_label = bench_id.replace("_", " ").title()
            lines.append(
                f"| {bench_label} | {rows_str} | "
                f"{cubecl_ms:.2f} | {cpu_ms:.2f} | {cudf_str} | {ratio_str} |"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare polars-cubecl vs cuDF benchmark results"
    )
    parser.add_argument(
        "--cudf-results",
        type=str,
        default=None,
        help="Path to cudf nvbench text output",
    )
    parser.add_argument(
        "--cubecl-dir",
        type=Path,
        default=Path("target") / "criterion",
        help="Path to criterion output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write markdown report to this file (default: stdout)",
    )
    args = parser.parse_args()

    cudf_data = parse_cudf_file(args.cudf_results) if args.cudf_results else None
    cubecl_data = parse_criterion_dir(args.cubecl_dir)

    if not cubecl_data:
        print("error: no criterion results found", file=sys.stderr)
        sys.exit(1)

    report = format_table(cubecl_data, cudf_data)

    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
