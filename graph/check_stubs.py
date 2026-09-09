"""
Check for junk/stub nodes in the vault.

Scans /data/vault for entity notes that look like descriptive phrases
rather than real named entities. Reports:
- Entities with entity_type "unknown" (stubs created by aggregate.py)
- Entity names that fail the proper-name heuristic
- Entity names that match known prompt-leakage patterns

Usage:
    python graph/check_stubs.py
"""

import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.schema import _is_proper_name, _PROMPT_LEAK_PATTERNS

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
VAULT_DIR = os.path.join(PROJECT_ROOT, "data", "vault")


def check_vault():
    if not os.path.exists(VAULT_DIR):
        print(f"Vault directory not found: {VAULT_DIR}")
        sys.exit(1)

    md_files = [f for f in os.listdir(VAULT_DIR) if f.endswith(".md")]
    if not md_files:
        print("No vault notes found.")
        sys.exit(0)

    print(f"Scanning {len(md_files)} vault notes...\n")

    problems = {
        "unknown_type": [],
        "descriptive_phrase": [],
        "prompt_leak": [],
    }

    for filename in sorted(md_files):
        filepath = os.path.join(VAULT_DIR, filename)
        entity_name = os.path.splitext(filename)[0]
        entity_type = "unknown"

        # Parse YAML frontmatter for entity_type
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                frontmatter = content[3:end]
                for line in frontmatter.strip().split("\n"):
                    if line.startswith("entity_type:"):
                        entity_type = line.split(":", 1)[1].strip()

        # Check for unknown entity type (stub node)
        if entity_type == "unknown":
            problems["unknown_type"].append(entity_name)

        # Check if name looks like a descriptive phrase
        if not _is_proper_name(entity_name):
            problems["descriptive_phrase"].append(entity_name)

        # Check for prompt leakage
        name_lower = entity_name.lower()
        for pattern in _PROMPT_LEAK_PATTERNS:
            if pattern in name_lower:
                problems["prompt_leak"].append((entity_name, pattern))
                break

    # Report
    total_problems = 0

    if problems["prompt_leak"]:
        print("=" * 60)
        print("[!] PROMPT LEAKAGE (instruction text echoed as entity names):")
        print("=" * 60)
        for name, pattern in problems["prompt_leak"]:
            print(f"  - \"{name}\" (matched: '{pattern}')")
        total_problems += len(problems["prompt_leak"])
        print()

    if problems["descriptive_phrase"]:
        print("=" * 60)
        print("[!] DESCRIPTIVE PHRASES (not proper-noun entity names):")
        print("=" * 60)
        for name in problems["descriptive_phrase"]:
            print(f"  - \"{name}\"")
        total_problems += len(problems["descriptive_phrase"])
        print()

    if problems["unknown_type"]:
        print("=" * 60)
        print("[?] UNKNOWN ENTITY TYPE (stub nodes, may be legitimate):")
        print("=" * 60)
        for name in problems["unknown_type"]:
            # Only flag if also a descriptive phrase
            marker = " <- JUNK" if name in problems["descriptive_phrase"] else ""
            print(f"  - \"{name}\"{marker}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total vault notes:         {len(md_files)}")
    print(f"Prompt leakage:            {len(problems['prompt_leak'])}")
    print(f"Descriptive phrases:       {len(problems['descriptive_phrase'])}")
    print(f"Unknown-type stubs:        {len(problems['unknown_type'])}")
    print(f"Total problems:            {total_problems}")

    if total_problems == 0:
        print("\n[OK] No junk nodes detected!")
    else:
        print(f"\n[FAIL] {total_problems} junk node(s) found. Review and re-extract.")

    return total_problems


if __name__ == "__main__":
    sys.exit(0 if check_vault() == 0 else 1)
