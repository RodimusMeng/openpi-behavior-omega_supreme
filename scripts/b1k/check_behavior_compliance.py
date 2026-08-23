"""Fail-closed checker for a BEHAVIOR 2026 compliance manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openpi.weaver import behavior_compliance


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--submission", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.manifest).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Compliance manifest must contain a JSON object")
    violations = behavior_compliance.validate_manifest(value, submission=args.submission)
    result = {
        "manifest": str(path),
        "submission": bool(args.submission),
        "compliant": not violations,
        "violations": [item.__dict__ for item in violations],
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if violations:
        raise SystemExit(2)
    return result


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
