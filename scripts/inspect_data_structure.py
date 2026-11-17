"""Describe the SHAPE of the restricted BAR-ATH bundle without printing any of its content.

Prints only structure: file sizes, row counts, XML tag/attribute NAMES (never their values or
text), field counts per record, and how often a field is empty. No coordinates, timestamps, tags,
names or matrix values are ever printed, so the output is safe to paste in an issue or a chat.

    python scripts/inspect_data_structure.py                 # uses MTAH_DATA_DIR or ./data
"""

import collections
import csv
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402


def xml_schema(path, max_depth=8):
    """Tag tree with counts and attribute NAMES. Text and attribute values are discarded."""
    counts = collections.Counter()
    attrs = collections.defaultdict(set)
    has_text = set()

    def walk(node, prefix, depth):
        key = prefix + "/" + node.tag
        counts[key] += 1
        attrs[key].update(node.attrib.keys())
        if node.text and node.text.strip():
            has_text.add(key)
        if depth < max_depth:
            for child in node:
                walk(child, key, depth + 1)

    walk(ET.parse(path).getroot(), "", 0)
    for key in sorted(counts, key=lambda k: (k.count("/"), k)):
        extra = []
        if attrs[key]:
            extra.append("attrs=" + ",".join(sorted(attrs[key])))
        if key in has_text:
            extra.append("has text")
        print(f"  {key}  x{counts[key]}  {' '.join(extra)}")


def atd_raw_shape(path):
    """Separators per line, entries per section, fields per entry, empty-field rates."""
    sections = collections.Counter()
    entries = collections.Counter()
    player_fields = collections.Counter()
    ball_fields = collections.Counter()
    ball_empty = n = 0
    with open(path) as handle:
        for line in handle:
            n += 1
            parts = line.rstrip("\n").split(":")
            sections[len(parts)] += 1
            if len(parts) >= 2:
                players = [p for p in parts[1].split(";") if p != ""]
                entries[len(players)] += 1
                for p in players:
                    player_fields[len(p.split(","))] += 1
            if len(parts) >= 3:
                fields = parts[2].split(",")
                ball_fields[len(fields)] += 1
                if all(f.strip() in ("", "NaN", "nan") for f in fields):
                    ball_empty += 1
    print(f"  lines: {n}")
    print(f"  ':'-sections per line: {dict(sections)}")
    print(f"  player entries per line (top 8): {entries.most_common(8)}")
    print(f"  fields per player entry: {dict(player_fields)}")
    print(f"  fields in ball section: {dict(ball_fields)}")
    print(f"  lines with an empty ball section: {ball_empty} ({ball_empty / max(n, 1):.1%})")


def csv_shape(path):
    with open(path) as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    print(f"  columns ({len(header)}): {header}")
    print(f"  rows: {rows}")


def main():
    paths = asset_paths()
    for key in ("atd_meta", "atd_raw", "smart_tagging", "homography", "video"):
        path = paths.get(key)
        if not path or not os.path.exists(path):
            print(f"[{key}] missing: {path}")
            continue
        print(f"[{key}] {os.path.basename(path)}  {os.path.getsize(path) / 1e6:.1f} MB")
        if key in ("atd_meta", "smart_tagging"):
            xml_schema(path)
        elif key == "atd_raw":
            atd_raw_shape(path)
        elif key == "homography":
            csv_shape(path)


if __name__ == "__main__":
    main()
