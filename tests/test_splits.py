from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from deepca.splits import (
    SplitError,
    canonicalize_case_id,
    load_resolved_splits,
    resolve_case_identifier,
)


class SplitTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def write_json(self, value: Any, name: str = "split.json") -> Path:
        path = self.directory / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    @staticmethod
    def minimal_splits(**overrides: Any) -> dict[str, Any]:
        document: dict[str, Any] = {
            "train": [1],
            "val": [2],
            "test": [3],
        }
        document.update(overrides)
        return document

    def test_prompt_path_examples_and_numeric_forms(self) -> None:
        self.assertEqual(
            canonicalize_case_id("/any/root/lca/1/prefix_02.npz", "lca"),
            "lca_0001",
        )
        self.assertEqual(
            canonicalize_case_id("/any/root/rca_0508.npz", "rca"),
            "rca_0508",
        )
        self.assertEqual(canonicalize_case_id(8, "rca"), "rca_0008")
        self.assertEqual(canonicalize_case_id(8.0, "rca"), "rca_0008")
        self.assertEqual(canonicalize_case_id("1.npz", "lca"), "lca_0001")
        self.assertEqual(canonicalize_case_id("rca_0007"), "rca_0007")

    def test_windows_paths_and_case_insensitive_vessel_tokens(self) -> None:
        self.assertEqual(
            canonicalize_case_id(
                r"C:\dataset\RCA\12\prefix_02.npz", "RCA"
            ),
            "rca_0012",
        )
        self.assertEqual(
            canonicalize_case_id(r"C:\dataset\LCA_0042.NPZ", "lca"),
            "lca_0042",
        )

    def test_more_than_four_digits_are_not_truncated(self) -> None:
        case = resolve_case_identifier("rca_12345.npz", "rca")
        self.assertEqual(case.case_number, 12345)
        self.assertEqual(case.canonical_id, "rca_12345")

    def test_top_level_aliases_preserve_order_and_provenance(self) -> None:
        document = {
            "training": [
                3,
                {"path": "/root/rca/1/prefix_02.npz"},
                {"sample_name": "rca_0002", "case_id": 2},
            ],
            "dev": [{"file": "rca_0004.npz"}],
            "testing": [5],
        }
        path = self.write_json(document)
        resolved = load_resolved_splits(path, "rca")

        self.assertEqual(resolved.train, ["rca_0003", "rca_0001", "rca_0002"])
        self.assertEqual(resolved.val, ["rca_0004"])
        self.assertEqual(resolved.test, ["rca_0005"])
        self.assertEqual(resolved.container_path, ())
        self.assertEqual(
            resolved.aliases,
            {"train": "training", "val": "dev", "test": "testing"},
        )
        self.assertEqual(resolved.provenance["train"][1].index, 1)
        self.assertEqual(
            resolved.provenance["train"][1].location, "$.training[1]"
        )
        self.assertEqual(
            [item.field for item in resolved.provenance["train"][2].identifiers],
            ["sample_name", "case_id"],
        )
        self.assertEqual(
            resolved.source_sha256,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        # Manifest output must remain directly serializable for experiment logs.
        json.dumps(resolved.as_manifest())

    def test_every_alias_and_recognized_nested_container(self) -> None:
        train_aliases = ("train", "training")
        val_aliases = ("val", "validation", "valid", "dev")
        test_aliases = ("test", "testing")
        counter = 0
        for train_alias in train_aliases:
            for val_alias in val_aliases:
                for test_alias in test_aliases:
                    with self.subTest(
                        train=train_alias, val=val_alias, test=test_alias
                    ):
                        counter += 1
                        split_mapping = {
                            train_alias: [1],
                            val_alias: [2],
                            test_alias: [3],
                        }
                        document = {
                            "dataset": {"partitions": {"splits": split_mapping}}
                        }
                        path = self.write_json(document, f"aliases_{counter}.json")
                        resolved = load_resolved_splits(path, "lca")
                        self.assertEqual(resolved.train, ["lca_0001"])
                        self.assertEqual(resolved.val, ["lca_0002"])
                        self.assertEqual(resolved.test, ["lca_0003"])
                        self.assertEqual(
                            resolved.container_path,
                            ("dataset", "partitions", "splits"),
                        )

    def test_record_fields_must_resolve_consistently(self) -> None:
        self.assertEqual(
            canonicalize_case_id(
                {
                    "path": "/root/rca/7/prefix_02.npz",
                    "sample_name": "rca_0007",
                    "case_number": 7,
                    "ignored_metadata": "anything",
                },
                "rca",
            ),
            "rca_0007",
        )
        with self.assertRaisesRegex(SplitError, "Conflicting case identifiers"):
            canonicalize_case_id(
                {"case_id": 7, "name": "rca_0008"}, "rca"
            )
        with self.assertRaisesRegex(SplitError, "no supported identifier"):
            canonicalize_case_id({"metadata": "rca_0007"}, "rca")

    def test_bare_bool_fraction_and_non_identifier_suffix_are_rejected(self) -> None:
        with self.assertRaisesRegex(SplitError, "Boolean"):
            canonicalize_case_id(True, "rca")
        with self.assertRaisesRegex(SplitError, "finite integer"):
            canonicalize_case_id(1.5, "rca")
        with self.assertRaisesRegex(SplitError, "requires a configured vessel"):
            canonicalize_case_id(1)
        with self.assertRaisesRegex(SplitError, "Could not resolve"):
            canonicalize_case_id("prefix_02.npz", "rca")

    def test_vessel_mismatch_and_ambiguous_path_are_rejected(self) -> None:
        with self.assertRaisesRegex(SplitError, "configured for RCA"):
            canonicalize_case_id("lca_0001.npz", "rca")
        with self.assertRaisesRegex(SplitError, "Ambiguous case identifier"):
            canonicalize_case_id("/root/lca/1/rca_0002.npz", "lca")

    def test_duplicate_after_alias_resolution_is_rejected(self) -> None:
        path = self.write_json(
            {
                "train": [1, "rca_0001.npz"],
                "val": [2],
                "test": [3],
            }
        )
        with self.assertRaisesRegex(SplitError, "Duplicate case rca_0001"):
            load_resolved_splits(path, "rca")

    def test_cross_split_leakage_after_path_resolution_is_rejected(self) -> None:
        path = self.write_json(
            {
                "train": [{"path": "/root/rca/1/prefix_02.npz"}],
                "validation": ["rca_0001.npz"],
                "testing": [2],
            }
        )
        with self.assertRaisesRegex(SplitError, "Cross-split leakage"):
            load_resolved_splits(path, "rca")

    def test_multiple_aliases_for_one_split_are_rejected(self) -> None:
        path = self.write_json(
            {
                "train": [1],
                "training": [2],
                "val": [3],
                "test": [4],
            }
        )
        with self.assertRaisesRegex(SplitError, "multiple aliases"):
            load_resolved_splits(path, "rca")

    def test_multiple_candidate_containers_are_rejected(self) -> None:
        path = self.write_json(
            {
                "splits": self.minimal_splits(),
                "partitions": {
                    "training": [4],
                    "validation": [5],
                    "testing": [6],
                },
            }
        )
        with self.assertRaisesRegex(SplitError, "Multiple candidate"):
            load_resolved_splits(path, "rca")

    def test_equivalent_root_and_nested_splits_use_richer_records(self) -> None:
        document = {
            "schema_version": 2,
            "train": ["/root/rca_0003.npz", "/root/rca_0001.npz"],
            "val": ["/root/rca_0004.npz"],
            "test": ["/root/rca_0002.npz"],
            "splits": {
                "training": [
                    {"case_name": "rca_0003", "path": "/root/rca_0003.npz"},
                    {"case_name": "rca_0001", "path": "/root/rca_0001.npz"},
                ],
                "validation": [
                    {"case_name": "rca_0004", "path": "/root/rca_0004.npz"}
                ],
                "testing": [
                    {"case_name": "rca_0002", "path": "/root/rca_0002.npz"}
                ],
            },
        }
        resolved = load_resolved_splits(self.write_json(document), "rca")

        self.assertEqual(resolved.container_path, ("splits",))
        self.assertEqual(resolved.train, ["rca_0003", "rca_0001"])
        self.assertEqual(
            resolved.aliases,
            {"train": "training", "val": "validation", "test": "testing"},
        )
        self.assertEqual(
            resolved.provenance["train"][0].location,
            "$.splits.training[0]",
        )
        self.assertEqual(
            [
                item.field
                for item in resolved.provenance["train"][0].identifiers
            ],
            ["case_name", "path"],
        )

    def test_lca_path_stem_case_names_use_the_physical_case_path(self) -> None:
        def record(case_number: int) -> dict[str, str]:
            path = f"/features/lca/{case_number}/prefix_02.npz"
            return {"case_name": "prefix_02", "path": path}

        document = {
            "schema_version": 2,
            "train": [
                "/features/lca/1/prefix_02.npz",
                "/features/lca/2/prefix_02.npz",
            ],
            "val": ["/features/lca/3/prefix_02.npz"],
            "test": ["/features/lca/4/prefix_02.npz"],
            "splits": {
                "train": [record(1), record(2)],
                "val": [record(3)],
                "test": [record(4)],
            },
        }
        resolved = load_resolved_splits(self.write_json(document), "lca")

        self.assertEqual(resolved.container_path, ("splits",))
        self.assertEqual(resolved.train, ["lca_0001", "lca_0002"])
        self.assertEqual(
            [
                item.field
                for item in resolved.provenance["train"][0].identifiers
            ],
            ["path"],
        )
        with self.assertRaisesRegex(SplitError, "Invalid record field 'case_name'"):
            canonicalize_case_id(
                {
                    "case_name": "different_prefix",
                    "path": "/features/lca/1/prefix_02.npz",
                },
                "lca",
            )

    def test_conflicting_root_and_nested_splits_remain_ambiguous(self) -> None:
        document = self.minimal_splits(
            splits={
                "train": [1],
                "val": [2],
                "test": [4],
            }
        )
        with self.assertRaisesRegex(SplitError, "Multiple candidate"):
            load_resolved_splits(self.write_json(document), "rca")

    def test_missing_split_and_non_list_split_are_rejected(self) -> None:
        missing = self.write_json(
            {"train": [1], "val": [2]}, "missing.json"
        )
        with self.assertRaisesRegex(SplitError, "missing 'test'"):
            load_resolved_splits(missing, "rca")

        non_list = self.write_json(
            {"train": [1], "val": [2], "test": 3}, "non_list.json"
        )
        with self.assertRaisesRegex(SplitError, "must be a JSON list"):
            load_resolved_splits(non_list, "rca")

    def test_duplicate_json_keys_are_rejected_before_resolution(self) -> None:
        path = self.directory / "duplicate_key.json"
        path.write_text(
            '{"train":[1],"train":[2],"val":[3],"test":[4]}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SplitError, "duplicate key"):
            load_resolved_splits(path, "rca")


if __name__ == "__main__":
    unittest.main()
