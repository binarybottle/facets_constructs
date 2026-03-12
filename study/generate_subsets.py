"""
generate_subsets.py
-------------------
Pre-generates balanced subsets of constructs for the pairwise study.

Each participant is assigned K of 90 constructs. For each construct in their
subset, either the canonical item name or one randomly chosen synonym is used
as the displayed term (never both in the same session).

Run this script locally whenever items.csv or design parameters change, then
deploy the resulting data/subsets.json alongside the study.

Usage:
    python generate_subsets.py [--k 26] [--n 400] [--seed 42]
"""

import argparse
import csv
import json
import random
from pathlib import Path


def load_items(csv_path):
    items = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Items', '').strip()
            if not name:
                continue
            synonyms = [
                row.get('Synonym 1', '').strip(),
                row.get('Synonym 2', '').strip(),
                row.get('Synonym 3', '').strip(),
            ]
            synonyms = [s for s in synonyms if s]
            items.append({'name': name, 'synonyms': synonyms})
    return items


def generate_balanced_subsets(items, k=26, n_subsets=400, seed=42):
    """
    Generate n_subsets balanced subsets of k constructs.

    Balance strategy: greedy minimum-frequency selection. At each subset, the
    k constructs with the lowest selection counts so far are preferentially
    chosen (with a small random perturbation to break ties).

    For each selected construct, the displayed term is chosen randomly:
    either the canonical item name or one of its synonyms.
    """
    rng = random.Random(seed)
    n_items = len(items)

    if k > n_items:
        raise ValueError(f"k={k} cannot exceed n_items={n_items}")

    item_counts = [0] * n_items
    subsets = []

    for subset_idx in range(n_subsets):
        # Score each item: lower count → higher priority; add noise to break ties
        scores = [item_counts[i] + rng.random() * 0.5 for i in range(n_items)]
        selected_indices = sorted(range(n_items), key=lambda i: scores[i])[:k]
        rng.shuffle(selected_indices)  # random ordering within subset

        for idx in selected_indices:
            item_counts[idx] += 1

        subset = []
        for construct_idx in selected_indices:
            item = items[construct_idx]
            # Randomly choose: canonical item (0) or one of its synonyms (1..n)
            n_syns = len(item['synonyms'])
            choice = rng.randint(0, n_syns)   # 0 = canonical, 1..n = synonym index+1

            if choice == 0:
                term = item['name']
                is_synonym = False
                synonym_index = None
            else:
                synonym_index = choice - 1
                term = item['synonyms'][synonym_index]
                is_synonym = True

            subset.append({
                'construct_index': construct_idx,
                'canonical_item':  item['name'],
                'term':            term,
                'is_synonym':      is_synonym,
                'synonym_index':   synonym_index,
            })

        subsets.append(subset)

    return subsets, item_counts


def check_balance(item_counts, n_items, k, n_subsets):
    target = n_subsets * k / n_items
    mn, mx = min(item_counts), max(item_counts)
    print(f"  Target frequency per construct: {target:.1f}")
    print(f"  Actual range: {mn}–{mx}  (spread: {mx - mn})")
    print(f"  Expected cross-construct pair coverage: "
          f"{n_subsets * (k*(k-1)//2) / (n_items*(n_items-1)//2):.1f} obs/pair")


def main():
    parser = argparse.ArgumentParser(description='Generate balanced subsets for pairwise study')
    parser.add_argument('--k',    type=int, default=26,  help='Constructs per participant (default: 26)')
    parser.add_argument('--n',    type=int, default=400, help='Number of subsets to generate (default: 400)')
    parser.add_argument('--seed', type=int, default=42,  help='Random seed (default: 42)')
    parser.add_argument('--items', default='data/items.csv', help='Path to items CSV')
    parser.add_argument('--out',   default='data/subsets.json', help='Output JSON path')
    args = parser.parse_args()

    items_path = Path(__file__).parent / args.items
    out_path   = Path(__file__).parent / args.out

    print(f"Loading items from {items_path} ...")
    items = load_items(items_path)
    print(f"  {len(items)} items loaded")

    print(f"\nGenerating {args.n} subsets of k={args.k} constructs (seed={args.seed}) ...")
    subsets, item_counts = generate_balanced_subsets(
        items, k=args.k, n_subsets=args.n, seed=args.seed
    )

    print("\nBalance check:")
    check_balance(item_counts, len(items), args.k, args.n)

    output = {
        'meta': {
            'n_items':              len(items),
            'k_per_participant':    args.k,
            'n_subsets':            args.n,
            'seed':                 args.seed,
            'pairs_per_participant': args.k * (args.k - 1) // 2,
        },
        'subsets': subsets,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(subsets)} subsets → {out_path}")
    print(f"Pairs per participant: {args.k * (args.k - 1) // 2}")
    print(f"Estimated time @ 3.5s/pair: "
          f"{args.k * (args.k - 1) // 2 * 3.5 / 60:.1f} min")


if __name__ == '__main__':
    main()
