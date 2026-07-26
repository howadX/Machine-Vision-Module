#!/usr/bin/env python3
"""Validate canonical MVM skills and require audits for public-contract drift."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPECTED_SKILLS = {
    "mvm-vision": {
        "peer": "../mvm-dataset/SKILL.md",
        "markers": (
            "snapshot",
            "recap",
            "inspect",
            "mvm run",
            "withheld",
            "unknown",
            "phantom",
            "using_object",
        ),
    },
    "mvm-dataset": {
        "peer": "../mvm-vision/SKILL.md",
        "markers": (
            "--record",
            "--no-vision-log",
            "mvm dataset check",
            "canonical observation",
            "session.toml",
            "provenance",
            "withheld",
            "unknown",
            "safe",
            "anonymous",
            "full",
        ),
    },
}

STALE_MARKERS = (
    "scrfd",
    "arcface 512",
    "hsemotion",
    "mediapipe holistic",
    "lightgbm",
    "mvm_bridge.py",
    "sight-journal",
)

PUBLIC_SURFACES = {
    "mvm/crates/mvm-cli/src/help.rs",
    "mvm/crates/mvm-cli/src/main.rs",
    "mvm/crates/mvm-cli/src/mcp.rs",
    "mvm/crates/mvm-cli/src/recap_cmd.rs",
    "mvm/crates/mvm-cli/src/dataset_cmd.rs",
    "mvm/crates/mvm-cli/src/render.rs",
    "mvm/crates/mvm-cli/src/state.rs",
}

PUBLIC_PREFIXES = ("mvm/crates/mvm-dataset/src/",)

SKILL_PATHS = {
    "mvm/skills/mvm-vision/SKILL.md",
    "mvm/skills/mvm-dataset/SKILL.md",
}

AUDIT_ARTIFACT = ".github/mvm-skill-audit.md"


def parse_frontmatter(text: str, path: Path) -> tuple[dict[str, str], str, list[str]]:
    errors: list[str] = []
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return {}, text, [f"{path}: frontmatter must start with ---"]
    try:
        closing = lines.index("---", 1)
    except ValueError:
        return {}, text, [f"{path}: frontmatter has no closing ---"]

    metadata: dict[str, str] = {}
    for line in lines[1:closing]:
        if ":" not in line:
            errors.append(f"{path}: invalid frontmatter line: {line!r}")
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key or not value:
            errors.append(f"{path}: empty frontmatter key or value")
        elif key in metadata:
            errors.append(f"{path}: duplicate frontmatter key {key!r}")
        else:
            metadata[key] = value
    return metadata, "\n".join(lines[closing + 1 :]), errors


def validate_tree(skills_root: Path) -> list[str]:
    errors: list[str] = []
    if not skills_root.is_dir():
        return [f"{skills_root}: canonical skills directory is missing"]

    actual_dirs = {path.name for path in skills_root.iterdir() if path.is_dir()}
    expected_dirs = set(EXPECTED_SKILLS)
    if actual_dirs != expected_dirs:
        errors.append(
            f"{skills_root}: skill directories must be exactly "
            f"{sorted(expected_dirs)}, got {sorted(actual_dirs)}"
        )
    loose_files = [path.name for path in skills_root.iterdir() if not path.is_dir()]
    if loose_files:
        errors.append(f"{skills_root}: unexpected files: {sorted(loose_files)}")

    for name, contract in EXPECTED_SKILLS.items():
        skill_dir = skills_root / name
        path = skill_dir / "SKILL.md"
        if not path.is_file():
            errors.append(f"{path}: required file is missing")
            continue
        extras = sorted(
            child.name for child in skill_dir.iterdir() if child.name != "SKILL.md"
        )
        if extras:
            errors.append(f"{skill_dir}: unexpected entries: {extras}")

        text = path.read_text(encoding="utf-8")
        metadata, body, parse_errors = parse_frontmatter(text, path)
        errors.extend(parse_errors)
        if set(metadata) != {"name", "description"}:
            errors.append(
                f"{path}: frontmatter keys must be exactly name, description; "
                f"got {sorted(metadata)}"
            )
        if metadata.get("name") != name:
            errors.append(
                f"{path}: frontmatter name must match directory {name!r}"
            )
        description = metadata.get("description", "")
        if len(description) > 300:
            errors.append(f"{path}: description exceeds 300 characters")
        if "use" not in description.lower():
            errors.append(f"{path}: description must state when to use the skill")
        if len(body.splitlines()) > 500:
            errors.append(f"{path}: body exceeds the 500-line portability limit")

        lowered = text.lower()
        for marker in contract["markers"]:
            if marker not in lowered:
                errors.append(f"{path}: required contract marker missing: {marker}")
        if contract["peer"].lower() not in lowered:
            errors.append(
                f"{path}: cross-skill link missing: {contract['peer']}"
            )
        for marker in STALE_MARKERS:
            if marker in lowered:
                errors.append(f"{path}: stale marker forbidden: {marker}")

    return errors


def audit_changed_paths(changed_paths: list[str]) -> list[str]:
    changed = set(changed_paths)
    public_changes = sorted(
        path
        for path in changed
        if path in PUBLIC_SURFACES
        or any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES)
    )
    if not public_changes:
        return []
    if changed & SKILL_PATHS or AUDIT_ARTIFACT in changed:
        return []
    rendered = ", ".join(public_changes)
    return [
        "public MVM contract changed without a skill update or explicit audit "
        f"artifact; human audit required: {rendered}. Update a canonical SKILL.md "
        f"or {AUDIT_ARTIFACT} in the same change."
    ]


def git_changed_paths(repo_root: Path, base: str, head: str) -> list[str]:
    if not base or set(base) == {"0"}:
        base = f"{head}^"
    command = ["git", "diff", "--name-only", f"{base}...{head}"]
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed: {result.stderr.strip() or 'unknown error'}"
        )
    return [line for line in result.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--base",
        help="git base revision for public-surface audit; omit for static checks",
    )
    parser.add_argument("--head", default="HEAD", help="git head revision")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    errors = validate_tree(repo_root / "mvm" / "skills")
    if args.base:
        try:
            changed = git_changed_paths(repo_root, args.base, args.head)
            errors.extend(audit_changed_paths(changed))
        except RuntimeError as exc:
            errors.append(str(exc))

    if errors:
        print("MVM skill contract gate FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    mode = "static + public-surface audit" if args.base else "static"
    print(f"MVM skill contract gate passed ({mode}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
