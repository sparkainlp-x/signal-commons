# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP.
# SPDX-License-Identifier: AGPL-3.0-only
import json
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from signal_commons import (
    analyze_frame,
    deduplicate_postcards,
    demo_frame,
    match_postcards,
)

PROJECT_DIR = Path(__file__).resolve().parent
SCRIPT = str(PROJECT_DIR / "signal_commons.py")
DEMO_FINGERPRINT = "sc-fp.v2:sha256:f6bbeddf13db0c16791a1ec460835478783f5381e17a420f7e1d9eecb4d613d9"


def hot_groups_frame(hot_count, hot=10.0, base=1.0):
    """Constant-valued frame with the first ``hot_count`` groups elevated."""
    values = [base] * 512
    for index in range(hot_count * 32):
        values[index] = hot
    return values


def run_cli(*args, stdin=None):
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        input=stdin,
        capture_output=True,
        text=True,
    )


class SignalCommonsTests(unittest.TestCase):
    def test_happy_path_emits_coarse_json_postcard(self):
        postcard = analyze_frame(demo_frame())
        encoded = json.dumps(postcard, allow_nan=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["frame"]["channels"], 512)
        self.assertEqual(decoded["frame"]["groups"], 16)
        self.assertEqual(decoded["summary"]["incident_class"], "localized")
        self.assertNotIn("residuals", decoded)
        self.assertNotIn("group_scores", decoded)
        self.assertEqual(decoded["fingerprint"]["version"], "sc-fp.v2")
        self.assertEqual(decoded["schema_version"], "signal-commons.postcard.v2")

    def test_postcard_contains_only_coarse_fields(self):
        postcard = analyze_frame(hot_groups_frame(3))
        self.assertEqual(set(postcard), {"schema_version", "frame", "summary", "fingerprint"})
        self.assertEqual(
            set(postcard["summary"]),
            {"incident_class", "severity_band", "affected_group_band", "pattern"},
        )
        for value in postcard["summary"].values():
            self.assertIsInstance(value, str)

    def test_pinned_demo_fingerprint(self):
        self.assertEqual(analyze_frame(demo_frame())["fingerprint"]["value"], DEMO_FINGERPRINT)

    def test_rejects_wrong_dimensions(self):
        for length in (0, 511, 513):
            with self.subTest(length=length):
                with self.assertRaisesRegex(ValueError, "exactly 512"):
                    analyze_frame([0.0] * length)

    def test_rejects_non_finite_values(self):
        for bad_value in (float("nan"), float("inf"), float("-inf"), 10**400):
            with self.subTest(value=bad_value):
                values = [0.0] * 512
                values[100] = bad_value
                with self.assertRaisesRegex(ValueError, "finite"):
                    analyze_frame(values)

    def test_rejects_bool_values(self):
        for bad_value in (True, False):
            with self.subTest(value=bad_value):
                values = [0.0] * 512
                values[7] = bad_value
                with self.assertRaisesRegex(ValueError, "index 7 must be a number"):
                    analyze_frame(values)

    def test_rejects_non_numeric_values_and_containers(self):
        values = [0.0] * 512
        values[3] = "1.0"
        with self.assertRaisesRegex(ValueError, "must be a number"):
            analyze_frame(values)
        for bad_frame in ("0" * 512, {"residuals": [0.0] * 512}, 5):
            with self.subTest(frame=type(bad_frame).__name__):
                with self.assertRaises(ValueError):
                    analyze_frame(bad_frame)

    def test_accepts_any_iterable_of_real_numbers(self):
        expected = analyze_frame(demo_frame())
        self.assertEqual(analyze_frame(tuple(demo_frame())), expected)
        self.assertEqual(analyze_frame(value for value in demo_frame()), expected)
        mixed = [Fraction(1, 1)] * 256 + [1] * 256
        self.assertEqual(analyze_frame(mixed)["summary"]["incident_class"], "quiet")

    def test_broad_elevation_is_widespread(self):
        for hot_count in (8, 9, 12, 16):
            with self.subTest(hot_count=hot_count):
                summary = analyze_frame(hot_groups_frame(hot_count))["summary"]
                self.assertEqual(summary["incident_class"], "widespread")
                self.assertEqual(summary["affected_group_band"], "8-16")
                self.assertEqual(summary["severity_band"], "very_high")

    def test_class_boundaries_match_count_bands(self):
        expected = {
            0: ("quiet", "0"),
            1: ("localized", "1"),
            2: ("distributed", "2-3"),
            3: ("distributed", "2-3"),
            4: ("distributed", "4-7"),
            7: ("distributed", "4-7"),
            8: ("widespread", "8-16"),
        }
        for hot_count, (incident_class, band) in expected.items():
            with self.subTest(hot_count=hot_count):
                summary = analyze_frame(hot_groups_frame(hot_count))["summary"]
                self.assertEqual(summary["incident_class"], incident_class)
                self.assertEqual(summary["affected_group_band"], band)

    def test_noise_rms_override_sets_absolute_baseline(self):
        uniform = [10.0] * 512
        self.assertEqual(analyze_frame(uniform)["summary"]["incident_class"], "widespread")
        self.assertEqual(analyze_frame(uniform, noise_rms=10.0)["summary"]["incident_class"], "quiet")
        for bad in (0.0, -1.0, float("nan"), float("inf"), True, "1.0"):
            with self.subTest(noise_rms=bad):
                with self.assertRaisesRegex(ValueError, "noise_rms"):
                    analyze_frame(uniform, noise_rms=bad)

    def test_small_bump_on_flat_frame_is_not_flagged(self):
        values = [1.0] * 512
        for index in range(32):
            values[index] = 1.2
        summary = analyze_frame(values)["summary"]
        self.assertEqual(summary["incident_class"], "quiet")
        self.assertEqual(summary["severity_band"], "none")

    def test_tiny_value_on_zero_frame_is_not_high_severity(self):
        values = [0.0] * 512
        values[5] = 1e-9
        summary = analyze_frame(values)["summary"]
        self.assertNotIn(summary["severity_band"], {"high", "very_high"})
        self.assertEqual(summary["incident_class"], "quiet")

    def test_exact_signature_matching_and_deduplication(self):
        quiet = analyze_frame([0.0] * 512)
        same_signature = analyze_frame([0.0] * 512)
        different_signature = analyze_frame(demo_frame())
        self.assertTrue(match_postcards(quiet, same_signature))
        self.assertFalse(match_postcards(quiet, different_signature))
        self.assertEqual(deduplicate_postcards([quiet, same_signature, different_signature]), [[0, 1]])

    def test_match_rejects_tampered_fingerprint(self):
        original = analyze_frame(demo_frame())
        tampered = json.loads(json.dumps(original))
        tampered["fingerprint"]["value"] = "sc-fp.v2:sha256:" + "0" * 64
        self.assertFalse(match_postcards(tampered, original))
        self.assertFalse(match_postcards(original, tampered))
        relabeled = json.loads(json.dumps(original))
        relabeled["summary"]["severity_band"] = "moderate"
        self.assertFalse(match_postcards(relabeled, original))
        old_version = json.loads(json.dumps(original))
        old_version["schema_version"] = "signal-commons.postcard.v1"
        self.assertFalse(match_postcards(old_version, original))

    def test_demo_cli_writes_valid_json(self):
        result = run_cli("--demo")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["schema_version"], "signal-commons.postcard.v2")
        self.assertEqual(payload["fingerprint"]["value"], DEMO_FINGERPRINT)
        self.assertEqual(result.stderr, "")

    def test_cli_input_file_object_and_bare_array(self):
        frame = hot_groups_frame(8)
        with tempfile.TemporaryDirectory() as tmp:
            for name, document in (("object.json", {"residuals": frame}), ("array.json", frame)):
                with self.subTest(document=name):
                    path = Path(tmp) / name
                    path.write_text(json.dumps(document), encoding="utf-8")
                    result = run_cli("--input", str(path))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    payload = json.loads(result.stdout)
                    self.assertEqual(payload, analyze_frame(frame))
                    self.assertEqual(payload["summary"]["incident_class"], "widespread")

    def test_cli_input_from_stdin(self):
        result = run_cli("--input", "-", stdin=json.dumps({"residuals": [0.0] * 512}))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["summary"]["incident_class"], "quiet")

    def test_cli_noise_rms_flag(self):
        stdin = json.dumps([10.0] * 512)
        default = json.loads(run_cli("--input", "-", stdin=stdin).stdout)
        self.assertEqual(default["summary"]["incident_class"], "widespread")
        scaled = json.loads(run_cli("--input", "-", "--noise-rms", "10", stdin=stdin).stdout)
        self.assertEqual(scaled["summary"]["incident_class"], "quiet")
        bad = run_cli("--demo", "--noise-rms", "0")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("noise", bad.stderr)

    def test_cli_rejects_invalid_input(self):
        result = run_cli("--input", "-", stdin=json.dumps([0.0] * 10))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("exactly 512", result.stderr)

    def test_cli_index_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "index.json"
            seen = []
            for _ in range(2):
                result = run_cli("--demo", "--index", str(index_path))
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["deduplication"]["scope"], "local_index")
                self.assertEqual(payload["postcard"]["fingerprint"]["value"], DEMO_FINGERPRINT)
                seen.append(payload["deduplication"]["duplicate_signature_seen"])
            self.assertEqual(seen, [False, True])
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(index["fingerprints"], [DEMO_FINGERPRINT])


if __name__ == "__main__":
    unittest.main()
