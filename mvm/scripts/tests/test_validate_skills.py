import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "validate_skills.py"
SPEC = importlib.util.spec_from_file_location("validate_skills", MODULE_PATH)
validate_skills = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validate_skills)


VISION_BODY = """---
name: mvm-vision
description: Use MVM snapshot, recap, or inspect to read a live or historical scene.
---

# MVM Vision

Use `snapshot`, then `recap`, then `inspect`, or `mvm_cortex`. Preserve `withheld` and `unknown`.
Treat a `phantom` as uncertain. Read `using_object`. Start `mvm run`.
The `current_cortex.json` entity key preserves incarnation semantics.
See [mvm-dataset](../mvm-dataset/SKILL.md).
"""

DATASET_BODY = """---
name: mvm-dataset
description: Use the MVM Dataset Factory to record, validate, retain, or export sessions.
---

# MVM Dataset

Use `mvm run --record` or `--no-vision-log`. Run `mvm dataset check`.
Read `Canonical Observation`, `session.toml`, `provenance`, `withheld`, and
`unknown`. Choose `safe`, `anonymous`, or `full` privacy.
See [mvm-vision](../mvm-vision/SKILL.md).
"""


class SkillTreeTests(unittest.TestCase):
    def make_tree(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        for name, body in (
            ("mvm-vision", VISION_BODY),
            ("mvm-dataset", DATASET_BODY),
        ):
            skill_dir = root / name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
        self.addCleanup(temp.cleanup)
        return root

    def test_valid_tree_passes(self):
        self.assertEqual(validate_skills.validate_tree(self.make_tree()), [])

    def test_extra_frontmatter_field_fails(self):
        root = self.make_tree()
        path = root / "mvm-vision" / "SKILL.md"
        path.write_text(
            VISION_BODY.replace("description:", "category: vision\ndescription:"),
            encoding="utf-8",
        )
        errors = validate_skills.validate_tree(root)
        self.assertTrue(any("frontmatter keys" in error for error in errors))

    def test_stale_python_stack_marker_fails(self):
        root = self.make_tree()
        path = root / "mvm-vision" / "SKILL.md"
        path.write_text(VISION_BODY + "\nUse SCRFD for faces.\n", encoding="utf-8")
        errors = validate_skills.validate_tree(root)
        self.assertTrue(any("stale marker" in error for error in errors))


class PublicSurfaceAuditTests(unittest.TestCase):
    def test_unrelated_change_needs_no_audit(self):
        self.assertEqual(validate_skills.audit_changed_paths(["README.md"]), [])

    def test_public_change_without_skill_audit_fails(self):
        errors = validate_skills.audit_changed_paths(
            ["mvm/crates/mvm-cli/src/mcp.rs"]
        )
        self.assertTrue(any("human audit required" in error for error in errors))

    def test_any_dataset_source_change_requires_audit(self):
        errors = validate_skills.audit_changed_paths(
            ["mvm/crates/mvm-dataset/src/router.rs"]
        )
        self.assertTrue(any("human audit required" in error for error in errors))

    def test_skill_change_satisfies_public_audit(self):
        self.assertEqual(
            validate_skills.audit_changed_paths(
                [
                    "mvm/crates/mvm-cli/src/mcp.rs",
                    "mvm/skills/mvm-vision/SKILL.md",
                ]
            ),
            [],
        )

    def test_explicit_audit_artifact_satisfies_public_audit(self):
        self.assertEqual(
            validate_skills.audit_changed_paths(
                [
                    "mvm/crates/mvm-dataset/src/observation.rs",
                    ".github/mvm-skill-audit.md",
                ]
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
