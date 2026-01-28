#!/usr/bin/env python3
import argparse
from collections import Counter
import pandas as pd


def parse_relation_count(relations_value: object) -> int:
    """Return how many relations are encoded in the space-separated string."""
    if pd.isna(relations_value):
        return 0
    text = str(relations_value).strip()
    if not text:
        return 0
    return len(text.split())


def load_counts(csv_path: str, group_by: str) -> Counter:
    """Load a rel_neighbors-style CSV and count relations per key.

    group_by: "head" groups by head id; "pair" groups by (head, tail).
    Each row is weighted by the number of relations listed in the Relations column.
    """
    df = pd.read_csv(csv_path)
    counts: Counter = Counter()

    for _, row in df.iterrows():
        weight = parse_relation_count(row.get("Relations"))
        if group_by == "head":
            key = int(row["Head"])
        else:
            key = (int(row["Head"]), int(row["Tail"]))
        counts[key] += weight

    return counts


def compute_differences(counts_a: Counter, counts_b: Counter) -> dict:
    """Compute counts_a - counts_b for all keys present in either mapping."""
    keys = set(counts_a.keys()) | set(counts_b.keys())
    return {key: counts_a.get(key, 0) - counts_b.get(key, 0) for key in keys}


def format_key(key, group_by: str) -> str:
    return f"{key[0]}->{key[1]}" if group_by == "pair" else str(key)


def print_extrema(diffs: dict, group_by: str, top: int) -> None:
    positives = sorted([(k, v) for k, v in diffs.items() if v > 0], key=lambda kv: kv[1], reverse=True)
    negatives = sorted([(k, v) for k, v in diffs.items() if v < 0], key=lambda kv: kv[1])
    zeros = sum(1 for _, v in diffs.items() if v == 0)

    if positives:
        print(f"Top {min(top, len(positives))} positive differences (first - second):")
        for key, value in positives[:top]:
            print(f"  {format_key(key, group_by)}: +{value}")
    else:
        print("No positive differences found.")

    if negatives:
        print(f"Top {min(top, len(negatives))} negative differences (first - second):")
        for key, value in negatives[:top]:
            print(f"  {format_key(key, group_by)}: {value}")
    else:
        print("No negative differences found.")

    print(f"Keys with zero difference: {zeros}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare weighted occurrences between two rel_neighbors-style CSV files. "
            "Counts are weighted by the number of relations per row."
        )
    )
    parser.add_argument("first_csv", help="Minuend CSV (counts from here minus second file)")
    parser.add_argument("second_csv", help="Subtrahend CSV")
    parser.add_argument(
        "--group-by",
        choices=["head", "pair"],
        default="head",
        help="Aggregate counts per head or per head-tail pair. Default: head.",
    )
    parser.add_argument("--top", type=int, default=5, help="How many extrema to display per side.")

    args = parser.parse_args()

    counts_first = load_counts(args.first_csv, args.group_by)
    counts_second = load_counts(args.second_csv, args.group_by)

    total_first = sum(counts_first.values())
    total_second = sum(counts_second.values())

    print(f"Total weighted relations in first CSV: {total_first}")
    print(f"Total weighted relations in second CSV: {total_second}")

    diffs = compute_differences(counts_first, counts_second)
    print_extrema(diffs, args.group_by, args.top)


if __name__ == "__main__":
    main()
