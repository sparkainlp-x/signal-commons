#!/usr/bin/env python3
# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP.
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline proof of concept for coarse classical telemetry incident postcards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import random
import re
import statistics
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

CHANNEL_COUNT = 512
GROUP_COUNT = 16
CHANNELS_PER_GROUP = 32
ROBUST_Z_THRESHOLD = 3.5
# Reference (expected) per-channel RMS of the residuals when nothing is wrong.
# The default assumes residuals already normalized to unit variance; override
# it with the ``noise_rms`` argument or the ``--noise-rms`` CLI flag.
DEFAULT_NOISE_RMS = 1.0
# For Gaussian noise with RMS sigma, the RMS of n independent samples has a
# standard deviation of roughly sigma / sqrt(2 * n). With n = 32 this is
# sigma / 8. It is used both as the lower bound on the robust scale and as the
# scale of the absolute-baseline check, so a group must stand out from the
# expected noise level, not merely from an unusually flat frame.
NOISE_SCALE_FRACTION = 1.0 / math.sqrt(2 * CHANNELS_PER_GROUP)
POSTCARD_SCHEMA_VERSION = "signal-commons.postcard.v2"
FINGERPRINT_VERSION = "sc-fp.v2"
INDEX_SCHEMA_VERSION = "signal-commons.index.v1"

_FEATURE_KEYS = (
    "incident_class",
    "severity_band",
    "affected_group_band",
    "pattern",
)
_ALLOWED_FEATURES = {
    "incident_class": {"quiet", "localized", "distributed", "widespread"},
    "severity_band": {"none", "moderate", "high", "very_high"},
    "affected_group_band": {"0", "1", "2-3", "4-7", "8-16"},
    "pattern": {"none", "one_group", "contiguous", "scattered"},
}
_FINGERPRINT_RE = re.compile(r"^sc-fp\.v2:sha256:[0-9a-f]{64}$")


def _rms(values: Sequence[float]) -> float:
    """Return a numerically stable root-mean-square for a non-empty group."""
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    return scale * math.sqrt(math.fsum((value / scale) ** 2 for value in values) / len(values))


def _affected_band(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 7:
        return "4-7"
    return "8-16"


def _incident_class(count: int) -> str:
    """Map a flagged-group count to a class aligned with the count bands."""
    if count == 0:
        return "quiet"
    if count == 1:
        return "localized"  # band "1"
    if count <= 7:
        return "distributed"  # bands "2-3" and "4-7"
    return "widespread"  # band "8-16"


def _severity(peak_z: float) -> str:
    if peak_z < 5.0:
        return "moderate"
    if peak_z < 8.0:
        return "high"
    return "very_high"


def _is_real_number(value: Any) -> bool:
    """Accept real numbers (including e.g. NumPy scalars), but not booleans."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def validate_noise_rms(noise_rms: Any) -> float:
    """Return the reference noise RMS as a positive finite float or raise ValueError."""
    if not _is_real_number(noise_rms):
        raise ValueError("noise_rms must be a real number")
    try:
        numeric = float(noise_rms)
    except (OverflowError, ValueError) as exc:
        raise ValueError("noise_rms must be a positive finite number") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("noise_rms must be a positive finite number")
    return numeric


def _pattern(indices: Sequence[int]) -> str:
    if not indices:
        return "none"
    if len(indices) == 1:
        return "one_group"
    if all(right == left + 1 for left, right in zip(indices, indices[1:])):
        return "contiguous"
    return "scattered"


def _features_from_summary(summary: Any) -> dict[str, str]:
    if not isinstance(summary, Mapping):
        raise ValueError("postcard summary must be an object")
    features: dict[str, str] = {}
    for key in _FEATURE_KEYS:
        value = summary.get(key)
        if not isinstance(value, str) or value not in _ALLOWED_FEATURES[key]:
            raise ValueError(f"invalid categorical feature: {key}")
        features[key] = value
    return features


def _fingerprint_for_features(features: Mapping[str, str]) -> dict[str, str]:
    canonical = {"version": FINGERPRINT_VERSION, "features": dict(features)}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return {
        "version": FINGERPRINT_VERSION,
        "algorithm": "sha256",
        "value": f"{FINGERPRINT_VERSION}:sha256:{digest}",
    }


def analyze_frame(
    residuals: Iterable[numbers.Real], *, noise_rms: float = DEFAULT_NOISE_RMS
) -> dict[str, Any]:
    """Analyze 512 finite residuals and return a postcard with no sample values.

    ``residuals`` may be any iterable of real numbers (lists, tuples, generators,
    NumPy arrays or scalars, ``fractions.Fraction``...). Booleans are rejected.
    ``noise_rms`` is the expected per-channel residual RMS for a healthy frame.

    A group is flagged when either check reaches ``ROBUST_Z_THRESHOLD``:

    * relative: (rms - median) / scale, with
      scale = max(1.4826 * MAD, NOISE_SCALE_FRACTION * noise_rms);
    * absolute: (rms - noise_rms) / (NOISE_SCALE_FRACTION * noise_rms).

    The absolute check still works when half or more of the groups are
    elevated (where the median itself moves into the elevated groups), and the
    noise-tied scale floor keeps flat or near-zero frames from turning tiny
    differences into large scores.
    """
    if isinstance(residuals, (str, bytes, bytearray, Mapping)) or not isinstance(residuals, Iterable):
        raise ValueError(f"residual frame must be an iterable of {CHANNEL_COUNT} numeric values")
    reference_rms = validate_noise_rms(noise_rms)
    items = list(residuals)
    if len(items) != CHANNEL_COUNT:
        raise ValueError(f"residual frame must contain exactly {CHANNEL_COUNT} values; got {len(items)}")

    values: list[float] = []
    for index, value in enumerate(items):
        if not _is_real_number(value):
            raise ValueError(f"residual at index {index} must be a number")
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"residual at index {index} must be finite") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"residual at index {index} must be finite")
        values.append(numeric)

    group_rms = [
        _rms(values[start : start + CHANNELS_PER_GROUP])
        for start in range(0, CHANNEL_COUNT, CHANNELS_PER_GROUP)
    ]
    center = statistics.median(group_rms)
    mad = statistics.median(abs(score - center) for score in group_rms)
    if mad > sys.float_info.max / 1.4826:
        mad_scale = sys.float_info.max
    else:
        mad_scale = mad * 1.4826
    # Expected spread of a group RMS under the reference noise model. It floors
    # the robust scale (MAD may be 0 on flat frames) and scales the absolute check.
    noise_scale = reference_rms * NOISE_SCALE_FRACTION
    scale = max(mad_scale, noise_scale)
    relative_z = [(score - center) / scale for score in group_rms]
    absolute_z = [(score - reference_rms) / noise_scale for score in group_rms]
    combined_z = [max(rel, ab) for rel, ab in zip(relative_z, absolute_z)]
    anomalous = [index for index, score in enumerate(combined_z) if score >= ROBUST_Z_THRESHOLD]
    count = len(anomalous)

    incident_class = _incident_class(count)
    severity = "none" if count == 0 else _severity(max(combined_z[index] for index in anomalous))

    summary = {
        "incident_class": incident_class,
        "severity_band": severity,
        "affected_group_band": _affected_band(count),
        "pattern": _pattern(anomalous),
    }
    # The summary is built here from the fixed category sets above, so it is
    # hashed directly; _features_from_summary is only needed to validate
    # untrusted postcards in _verified_fingerprint.
    return {
        "schema_version": POSTCARD_SCHEMA_VERSION,
        "frame": {
            "kind": "classical_residual",
            "channels": CHANNEL_COUNT,
            "groups": GROUP_COUNT,
            "channels_per_group": CHANNELS_PER_GROUP,
        },
        "summary": summary,
        "fingerprint": _fingerprint_for_features(summary),
    }


def _verified_fingerprint(postcard: Any) -> str | None:
    if not isinstance(postcard, Mapping) or postcard.get("schema_version") != POSTCARD_SCHEMA_VERSION:
        return None
    try:
        expected = _fingerprint_for_features(_features_from_summary(postcard.get("summary")))
    except ValueError:
        return None
    supplied = postcard.get("fingerprint")
    if supplied != expected:
        return None
    return expected["value"]


def match_postcards(left: Any, right: Any) -> bool:
    """Return whether two valid postcards have the same coarse v2 signature."""
    left_fingerprint = _verified_fingerprint(left)
    return left_fingerprint is not None and left_fingerprint == _verified_fingerprint(right)


def deduplicate_postcards(postcards: Sequence[Any]) -> list[list[int]]:
    """Return index groups for repeated, internally valid coarse signatures."""
    grouped: dict[str, list[int]] = {}
    for index, postcard in enumerate(postcards):
        fingerprint = _verified_fingerprint(postcard)
        if fingerprint is not None:
            grouped.setdefault(fingerprint, []).append(index)
    return [indices for indices in grouped.values() if len(indices) > 1]


def demo_frame() -> list[float]:
    """Build a reproducible synthetic frame with one elevated group."""
    rng = random.Random(29)
    values = [rng.gauss(0.0, 1.0) for _ in range(CHANNEL_COUNT)]
    for index in range(6 * CHANNELS_PER_GROUP, 7 * CHANNELS_PER_GROUP):
        values[index] += 7.5
    return values


def _read_residuals(path: str) -> Any:
    if path == "-":
        document = json.load(sys.stdin)
    else:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    if isinstance(document, Mapping):
        if "residuals" not in document:
            raise ValueError("JSON object input must contain a 'residuals' array")
        document = document["residuals"]
    return document


def _load_index(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"index file is not valid JSON: {path}") from exc
    if not isinstance(document, Mapping) or document.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError("index file has an unsupported schema")
    fingerprints = document.get("fingerprints")
    if not isinstance(fingerprints, list) or any(
        not isinstance(value, str) or not _FINGERPRINT_RE.fullmatch(value) for value in fingerprints
    ):
        raise ValueError(f"index file contains invalid or non-{FINGERPRINT_VERSION} fingerprints")
    return set(fingerprints)


def _write_index(path: Path, fingerprints: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = handle.name
            json.dump(
                {"schema_version": INDEX_SCHEMA_VERSION, "fingerprints": sorted(fingerprints)},
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _record_in_index(path: Path, fingerprint: str) -> bool:
    known = _load_index(path)
    duplicate = fingerprint in known
    if not duplicate:
        known.add(fingerprint)
        _write_index(path, known)
    return duplicate


def _noise_rms_arg(text: str) -> float:
    try:
        return validate_noise_rms(float(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid noise RMS {text!r}: must be a positive finite number") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", metavar="PATH", help="JSON array/object of residuals; use - for stdin")
    source.add_argument("--demo", action="store_true", help="analyze a reproducible synthetic frame (default)")
    parser.add_argument("--index", metavar="PATH", help="optional local JSON fingerprint index for exact-signature dedup")
    parser.add_argument(
        "--noise-rms",
        metavar="RMS",
        type=_noise_rms_arg,
        default=DEFAULT_NOISE_RMS,
        help=f"expected per-channel residual RMS of a healthy frame (default {DEFAULT_NOISE_RMS})",
    )
    args = parser.parse_args(argv)

    try:
        residuals = _read_residuals(args.input) if args.input else demo_frame()
        postcard = analyze_frame(residuals, noise_rms=args.noise_rms)
        if args.index:
            duplicate = _record_in_index(Path(args.index), postcard["fingerprint"]["value"])
            output: Any = {
                "postcard": postcard,
                "deduplication": {
                    "duplicate_signature_seen": duplicate,
                    "scope": "local_index",
                },
            }
        else:
            output = postcard
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError) as exc:
        print(f"signal_commons: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
