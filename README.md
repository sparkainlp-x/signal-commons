# Signal Commons

[![tests](https://github.com/sparkainlp-x/signal-commons/actions/workflows/tests.yml/badge.svg)](https://github.com/sparkainlp-x/signal-commons/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23049349.svg)](https://doi.org/10.5281/zenodo.23049349)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#workflow-and-mvp-scope)
[![Evidence: SYNTHETIC](https://img.shields.io/badge/evidence-SYNTHETIC-blue.svg)](#workflow-and-mvp-scope)

Signal Commons explores a narrow sharing problem: labs may want to compare telemetry anomalies without exchanging raw residual streams. This offline proof of concept takes a **classical 512-channel residual frame**, divides it into **16 groups of 32 channels**, and emits a small categorical incident postcard. It implements no Internet service or cross-lab transport.

## Workflow and MVP scope

1. Supply a JSON array of 512 numbers (or an object containing a `residuals` array), or run the reproducible demo frame. From Python, `analyze_frame` accepts any iterable of real numbers (lists, tuples, generators, NumPy arrays or scalars); booleans are rejected.
2. The analyzer checks the length, numeric types, and finiteness of every value, then computes each group's RMS. Each group gets two scores, and it is flagged if either reaches the prototype threshold of 3.5:
   - **Relative (robust z):** `(group RMS − median) / scale`, where `scale = max(1.4826 × MAD, noise_rms / √64)`. The median and median absolute deviation (MAD) are taken over the 16 group RMS values.
   - **Absolute baseline:** `(group RMS − noise_rms) / (noise_rms / √64)`. Here `noise_rms` is the expected per-channel residual RMS of a healthy frame. The default is `1.0`, meaning unit-variance residuals; override it with `--noise-rms`.

   `noise_rms / √64` is the approximate standard deviation of the RMS of 32 independent Gaussian samples with RMS `noise_rms` (`σ/√(2n)`, n = 32). This one quantity does two jobs:
   - It floors the robust scale, so a flat frame (MAD = 0) or a near-zero frame cannot turn tiny absolute differences into high scores.
   - It scales the absolute check, which still detects broad elevation when 8 or more groups are raised and the median moves into them.

   Groups below the baseline are never flagged.
3. It reduces the result to broad categories—incident class, severity band, affected-group-count band, and a coarse arrangement pattern. Raw samples, group scores, and group indices are not included in the postcard.
4. It hashes those categories with a versioned SHA-256 fingerprint. Optional local-index mode records only fingerprints and reports whether that same coarse signature was already seen in that index.

Category boundaries (scheme `sc-fp.v2` / `signal-commons.postcard.v2`):

| Flagged groups | `affected_group_band` | `incident_class` |
| --- | --- | --- |
| 0 | `0` | `quiet` |
| 1 | `1` | `localized` |
| 2–3 | `2-3` | `distributed` |
| 4–7 | `4-7` | `distributed` |
| 8–16 | `8-16` | `widespread` |

The severity band is taken from the highest combined score among flagged groups: `moderate` below 5, `high` from 5 to below 8, and `very_high` at 8 or above (`none` when nothing is flagged). The pattern is `one_group`, `contiguous`, or `scattered` according to the flagged group positions, which are themselves not reported.

Version 2 changes the categorical scheme: it adds the absolute-baseline check and the noise-tied scale floor (replacing the old `max(0.05 × median, 1e-12)` fallback) and realigns the class boundaries. So v2 fingerprints are not comparable with v1 fingerprints, and v1 postcards do not verify under v2. Postcards produced with different `--noise-rms` settings are not directly comparable either, because the setting is not recorded in the card.

The thresholds, the default noise RMS, and the category boundaries are illustrative heuristics, not calibrated detector settings. The fingerprint is useful for a repeatable demo of coarse signature matching, not for identifying a unique event. Matching is exact at the categorical-signature level, so distinct events can intentionally map to the same fingerprint.

## Quickstart

Requires Python 3.10 or later; the implementation uses only the Python standard library.

```bash
python3 signal_commons.py --demo
python3 -m unittest -v test_signal_commons.py
python3 -m py_compile signal_commons.py test_signal_commons.py
```

To make a quiet sample file and analyze it:

```bash
python3 -c 'import json; print(json.dumps({"residuals": [0.0] * 512}))' > frame.json
python3 signal_commons.py --input frame.json
```

A bare JSON array is also accepted. Use `--input -` to read JSON from standard input. If your residuals are not normalized to unit variance, pass their expected healthy per-channel RMS, for example `--noise-rms 0.02`. It must be a positive finite number. Invalid dimensions or values produce an error on stderr and a nonzero exit status; successful stdout is JSON.

To demonstrate local signature deduplication, run the same deterministic demo twice:

```bash
python3 signal_commons.py --demo --index ./local-index.json
python3 signal_commons.py --demo --index ./local-index.json
```

The first run reports `duplicate_signature_seen: false`; the second reports `true`. The index is an optional local JSON file containing fingerprints only. It is not a synchronized database, has no concurrent-writer coordination, and does not establish that two matching cards describe the same event. Index files only accept `sc-fp.v2` fingerprints. An index written by the v1 prototype is rejected with an error, so start a new index file.

## Example postcard

This is the JSON emitted by the built-in synthetic demo. The demo injects an elevated residual pattern into one group; this is generated test data, not a measured lab result.

```json
{
  "fingerprint": {
    "algorithm": "sha256",
    "value": "sc-fp.v2:sha256:f6bbeddf13db0c16791a1ec460835478783f5381e17a420f7e1d9eecb4d613d9",
    "version": "sc-fp.v2"
  },
  "frame": {
    "channels": 512,
    "channels_per_group": 32,
    "groups": 16,
    "kind": "classical_residual"
  },
  "schema_version": "signal-commons.postcard.v2",
  "summary": {
    "affected_group_band": "1",
    "incident_class": "localized",
    "pattern": "one_group",
    "severity_band": "very_high"
  }
}
```

## Safety, privacy, and limitations

- This prototype summarizes the described **local classical telemetry** only. It provides **no evidence of quantum effects** and makes no claim about a quantum mechanism.
- The postcard omits raw values, per-group scores, group identifiers, and timestamps. That is data minimization, not a privacy guarantee: the small set of categorical combinations can be guessed or enumerated, and repeated fingerprints can reveal that cards share a pattern. Treat fingerprints as potentially sensitive.
- SHA-256 here is an unkeyed, public hash, **not** a signature, access control, or proof of origin. This code has no networking, API, authentication, encryption-at-rest, consent flow, or production storage controls.
- It has not been validated with real lab data and makes no claim of uniqueness, security, detection quality, or field performance. The threshold can behave poorly under changing baselines, calibration differences, or nonstationary noise; the reported bands should not be used for operational decisions.
- Before any production or cross-lab deployment, require privacy/security review, informed consent and data-governance controls, public-key signing and verification, API/authentication and transport/storage protections, and validation with representative real-lab data. Version changes to the categorical scheme should use a new fingerprint version.

## Pilot metrics to evaluate

A real, consented pilot should measure alert precision and recall against independently adjudicated events; false-alert rate on quiet and shifted-baseline frames; agreement across labs for the same known event; accidental coarse-signature match rates for unrelated events; sensitivity to calibration and operating-regime changes; and whether participants can interpret the postcard without access to raw samples. These are proposed measurements, not results from this prototype.

## License

This software is available under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE).

Organizations that want to use it in proprietary products or services without AGPL obligations can contact the author about a commercial license via https://sparkainlpx.xyz.
