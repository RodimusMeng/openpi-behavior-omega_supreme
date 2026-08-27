"""Validate a BEHAVIOR v3.9.1 submission bundle without modifying it."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from openpi.weaver import submission_bundle


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="Unpacked submission bundle root")
    parser.add_argument(
        "--profile",
        choices=tuple(submission_bundle.PROFILES),
        default="docs_public_report",
        help="Explicitly select the challenge-docs or score_utils public-instance convention",
    )
    parser.add_argument(
        "--full-submission",
        action="store_true",
        help="Require 100 tasks and complete task x profile-instance x rollout coverage",
    )
    parser.add_argument("--manifest-name", default="manifest.json")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle = Path(args.bundle).expanduser().resolve()
    violations = submission_bundle.validate_submission_bundle(
        bundle,
        profile=args.profile,
        full_submission=bool(args.full_submission),
        manifest_name=args.manifest_name,
    )
    result = {
        "bundle": str(bundle),
        "profile": args.profile,
        "full_submission": bool(args.full_submission),
        "compliant": not violations,
        "claim_scope": (
            "Static bundle structure, declared revisions/configuration, and current artifact hash consistency only; "
            "this does not prove that an untampered official evaluator generated the artifacts."
        ),
        "violations": [dataclasses.asdict(item) for item in violations],
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if violations:
        raise SystemExit(2)
    return result


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
