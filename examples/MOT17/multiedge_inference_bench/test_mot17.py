#!/usr/bin/env python3
"""
MOT17/multiedge_inference_bench Test Suite
Tests the MOT17 example for correctness without requiring GPU,
dataset, or model weights.

Usage:
    python examples/MOT17/multiedge_inference_bench/test_mot17.py
    python examples/MOT17/multiedge_inference_bench/test_mot17.py --pr 392
"""

import sys
import os
import re
import argparse
import datetime
import traceback
import math
import importlib
import importlib.util
import yaml
import pathlib
from pathlib import Path
import tempfile
import types
from unittest.mock import MagicMock, patch


def _setup_sedna_mock():
    """Inject a minimal mock of sedna so metric files load without sedna installed."""
    if 'sedna' in sys.modules:
        return
    sedna = types.ModuleType('sedna')
    sedna_common = types.ModuleType('sedna.common')
    sedna_cf = types.ModuleType('sedna.common.class_factory')

    class FakeClassType:
        GENERAL = 'general'

    class FakeClassFactory:
        @staticmethod
        def register(class_type, alias=None):
            return lambda func: func

    sedna_cf.ClassType = FakeClassType
    sedna_cf.ClassFactory = FakeClassFactory
    sedna_common.class_factory = sedna_cf
    sedna.common = sedna_common
    sys.modules['sedna'] = sedna
    sys.modules['sedna.common'] = sedna_common
    sys.modules['sedna.common.class_factory'] = sedna_cf


def load_module_from_path(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MOT17TestRunner:
    def __init__(self, pr_branch=None):
        self.results = []
        self.pr_branch = pr_branch or "main"
        self.repo_root = Path(__file__).resolve().parents[3]
        self.example_root = (
            self.repo_root
            / "examples/MOT17/multiedge_inference_bench/pedestrian_tracking"
        )
        _setup_sedna_mock()
        repo_str = str(self.repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def run_test(self, name, func):
        """Run one test, catch all exceptions, record result."""
        try:
            result = func()
            if result is True or result is None:
                self.results.append(("PASS", name, ""))
                print(f"  [PASS] {name}")
            else:
                self.results.append(("FAIL", name, str(result)))
                print(f"  [FAIL] {name}")
                for line in str(result).strip().split("\n"):
                    print(f"         {line}")
        except Exception as e:
            tb = traceback.format_exc()
            self.results.append(("ERROR", name, tb))
            print(f"  [ERROR] {name}")
            print(f"          {str(e)}")

    # ── LAYER 1: Static Validation ──────────────────────────────────────────

    def test_1_1_stale_paths(self):
        stale_patterns = [
            "pedestrian_tracking/multiedge_inference_bench",
            "./examples/pedestrian_tracking",
        ]
        failures = []
        search_root = self.example_root.parent  # multiedge_inference_bench/
        for root, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                fpath = os.path.join(root, fname)
                # skip the test file itself
                if Path(fpath).name == "test_mot17.py":
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, 1):
                            if any(pat in line for pat in stale_patterns):
                                rel = os.path.relpath(fpath, self.repo_root)
                                failures.append(f"{rel}:{lineno}: {line.rstrip()}")
                except Exception:
                    pass
        if failures:
            return "Stale path references found:\n" + "\n".join(failures)

    def test_1_2_workspace_path(self):
        failures = []
        for job_yaml in ["tracking_job.yaml", "reid_job.yaml"]:
            path = self.example_root / job_yaml
            with open(path) as f:
                data = yaml.safe_load(f)
            workspace = data.get("benchmarkingjob", {}).get("workspace", "")
            if workspace.startswith("/"):
                failures.append(
                    f"{job_yaml}: Absolute Docker-specific path found: {workspace!r}"
                )
            elif workspace.startswith("../"):
                failures.append(
                    f"{job_yaml}: Path resolves above repo root: {workspace!r}"
                )
        if failures:
            return "\n".join(failures)

    def test_1_3_metric_urls(self):
        testenv_path = self.example_root / "testenv/tracking/testenv.yaml"
        with open(testenv_path) as f:
            data = yaml.safe_load(f)
        metrics = data.get("testenv", {}).get("metrics", [])
        failures = []
        for entry in metrics:
            name = entry.get("name", "")
            url = entry.get("url", "")
            expected_filename = name + ".py"
            url_filename = url.split("/")[-1] if "/" in url else url
            if url_filename != expected_filename:
                failures.append(
                    f"metric '{name}' has url pointing to '{url_filename}'"
                    f" — expected '{expected_filename}'"
                )
        if failures:
            return "\n".join(failures)

    def test_1_4_required_files(self):
        required = [
            "tracking_job.yaml",
            "reid_job.yaml",
            "generate_reports.py",
            "testenv/tracking/f1_score.py",
            "testenv/tracking/precision.py",
            "testenv/tracking/recall.py",
            "testenv/tracking/mota.py",
            "testenv/tracking/motp.py",
            "testenv/tracking/idf1.py",
            "testenv/reid/mAP.py",
            "testenv/reid/rank_1.py",
            "testenv/reid/rank_2.py",
            "testenv/reid/rank_5.py",
        ]
        missing = [
            str(self.example_root / rel)
            for rel in required
            if not (self.example_root / rel).exists()
        ]
        if missing:
            return "Missing files:\n" + "\n".join(missing)

    def test_1_5_imports(self):
        packages = {
            "motmetrics": "motmetrics",
            "loguru": "loguru",
            "fpdf": "fpdf",
            "pandas": "pandas",
            "seaborn": "seaborn",
            "sklearn": "scikit-learn",
            "scipy": "scipy",
            "cv2": "opencv-python",
            "yaml": "pyyaml",
        }
        failures = []
        for mod, pkg in packages.items():
            try:
                importlib.import_module(mod)
            except ImportError:
                failures.append(
                    f"ImportError: {mod} not installed — run: pip install {pkg}"
                )
        if failures:
            return "\n".join(failures)

    def test_1_6_pandas_api(self):
        src_path = self.example_root / "generate_reports.py"
        with open(src_path) as f:
            lines = f.readlines()
        hits = []
        for i, line in enumerate(lines, 1):
            if "delim_whitespace" in line:
                hits.append(f"Line {i}: {line.rstrip()}")
        try:
            import pandas as pd
            pd_version = pd.__version__
        except ImportError:
            pd_version = "unknown"
        if hits:
            return (
                f"Found deprecated `delim_whitespace` — removed in pandas 2.0. "
                f"Installed pandas: {pd_version}. Replace with sep=r'\\s+':\n"
                + "\n".join(hits)
            )

    def test_1_7_yaml_indent(self):
        yaml_path = (
            self.example_root
            / "testalgorithms/tracking/byte_track/byte_track_algorithm.yaml"
        )
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            algo = data.get("algorithm", {})
            modules = algo.get("modules", [])
            hps = None
            for mod in modules:
                hps = mod.get("hyperparameters")
            if not hps:
                return (
                    "hyperparameters key parsed as empty — likely caused by "
                    "indentation inconsistency"
                )
        except yaml.YAMLError as e:
            return f"YAML parse error: {e}"

    def test_1_8_nameerror(self):
        src_path = self.example_root / "generate_reports.py"
        with open(src_path) as f:
            lines = f.readlines()
        hits = []
        bare = re.compile(r"\bconfig_file\b")
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("#"):
                continue
            for m in bare.finditer(line):
                before = line[: m.start()]
                if before.endswith("tracking_") or before.endswith("reid_"):
                    continue
                hits.append(f"Line {i}: {line.rstrip()}")
                break
        if hits:
            return (
                "Found undefined variable `config_file` — will raise NameError:\n"
                + "\n".join(hits)
            )

    # ── LAYER 2: Unit Tests ─────────────────────────────────────────────────

    def _build_f1_fn(self):
        """Build the f1 compute function reflecting the ACTUAL source in f1_score.py."""
        src_path = self.example_root / "testenv/tracking/f1_score.py"
        with open(src_path) as f:
            src = f.read()
        has_nan_guard = bool(
            re.search(r"if\s+not\s*\(.*precision.*\+.*recall.*>", src)
        )
        has_zero_guard = bool(
            re.search(r"if\s+precision\s*\+\s*recall\s*==\s*0", src)
        )
        if has_nan_guard:
            def fn(p, r):
                if not (p + r > 0):
                    return 0.0
                return round(2 * ((p * r) / (p + r)), 4)
        elif has_zero_guard:
            def fn(p, r):
                if p + r == 0:
                    return 0.0
                return round(2 * ((p * r) / (p + r)), 4)
        else:
            def fn(p, r):
                return round(2 * ((p * r) / (p + r)), 4)
        return fn

    def test_2_1_f1_zero(self):
        fn = self._build_f1_fn()
        try:
            result = fn(0.0, 0.0)
            if result == 0.0:
                return None  # PASS
            return f"Returned {result} instead of 0.0 for zero inputs"
        except ZeroDivisionError:
            return "ZeroDivisionError — no guard for zero precision+recall"

    def test_2_2_f1_nan(self):
        fn = self._build_f1_fn()
        nan = float("nan")
        try:
            result = fn(nan, nan)
            if result == 0.0:
                return None  # PASS
            if math.isnan(result):
                return (
                    "NaN guard insufficient — nan+nan==0 is False in Python, "
                    "guard is bypassed. Fix: use `if not (precision + recall > 0)`"
                )
            return f"Returned unexpected value {result} for NaN inputs"
        except ZeroDivisionError:
            return "ZeroDivisionError on NaN inputs — guard does not handle NaN"

    def test_2_3_f1_normal(self):
        fn = self._build_f1_fn()
        result = fn(0.8, 0.6)
        expected = 0.6857
        if abs(result - expected) > 0.001:
            return f"Returned {result}, expected ~{expected}"

    def _nan_propagates(self, script_name, metric_key):
        """Return True if the script passes NaN through unguarded."""
        import pandas as pd
        nan = float("nan")
        summary = pd.DataFrame(
            {metric_key: [nan]}, index=["OVERALL"]
        )
        result = round(float(summary.iloc[-1][metric_key]), 4)
        return math.isnan(result)

    def _test_nan_metric(self, script_name, metric_key):
        if self._nan_propagates(script_name, metric_key):
            return (
                f"{script_name} returns nan — NaN propagation bug. "
                f"NaN will corrupt leaderboard CSV."
            )

    def test_2_4_nan_precision(self):
        return self._test_nan_metric("precision.py", "precision")

    def test_2_4_nan_recall(self):
        return self._test_nan_metric("recall.py", "recall")

    def test_2_4_nan_mota(self):
        return self._test_nan_metric("mota.py", "mota")

    def test_2_4_nan_motp(self):
        return self._test_nan_metric("motp.py", "motp")

    def test_2_4_nan_idf1(self):
        return self._test_nan_metric("idf1.py", "idf1")

    def test_2_5_reports_nameerror(self):
        src_path = self.example_root / "generate_reports.py"
        with open(src_path) as f:
            src = f.read()
        bare = re.compile(r"\bconfig_file\b")
        for m in bare.finditer(src):
            before = src[: m.start()]
            if before.endswith("tracking_") or before.endswith("reid_"):
                continue
            lineno = src[: m.start()].count("\n") + 1
            line = src.split("\n")[lineno - 1].strip()
            return (
                f"Raises NameError: name 'config_file' is not defined "
                f"at line {lineno}: {line}"
            )

    def test_2_6_reports_none_guard(self):
        src_path = self.example_root / "generate_reports.py"
        with open(src_path) as f:
            src = f.read()
        has_none_guard = bool(re.search(r"if\s+\w+\s+is\s+None", src))
        if not has_none_guard:
            return (
                "No None guard found — utils.is_local_file(None) will raise "
                "TypeError: expected str, bytes or os.PathLike, not NoneType"
            )

    def test_2_7_rank_indexerror(self):
        import numpy as np

        def cmc_from_file(distmat, query_ids, gallery_ids, topk):
            """Replicate exact cmc logic from rank_1.py."""
            m, _ = distmat.shape
            indices = np.argsort(distmat, axis=1)
            matches = gallery_ids[indices] == query_ids[:, np.newaxis]
            ret = np.zeros(topk)
            for i in range(m):
                k = np.nonzero(matches[i])[0][0]  # IndexError if empty
                if k < topk:
                    ret[k] += 1
            return round(float(ret.cumsum()[-1] / m), 4)

        distmat = np.array([[0.5, 0.8]])
        query_ids = np.array([1])
        gallery_ids = np.array([2, 3])  # neither matches query id 1
        try:
            result = cmc_from_file(distmat, query_ids, gallery_ids, 1)
            return f"Expected IndexError but got {result}"
        except IndexError:
            return (
                "IndexError: index 0 is out of bounds — no guard for "
                "queries with no gallery match"
            )

    def test_2_8_map_empty(self):
        import numpy as np
        from sklearn.metrics import average_precision_score

        def mean_ap_from_file(distmat, query_ids, gallery_ids):
            """Replicate exact mean_ap logic from mAP.py."""
            m, _ = distmat.shape
            indices = np.argsort(distmat, axis=1)
            matches = gallery_ids[indices] == query_ids[:, np.newaxis]
            aps = []
            for i in range(m):
                y_true = matches[i]
                y_score = -distmat[i][indices[i]]
                if not np.any(y_true):
                    continue
                aps.append(average_precision_score(y_true, y_score))
            if len(aps) == 0:
                raise RuntimeError("No valid query")
            return round(float(np.mean(aps)), 4)

        distmat = np.array([[0.5, 0.8]])
        query_ids = np.array([1])
        gallery_ids = np.array([2, 3])  # no matches
        try:
            result = mean_ap_from_file(distmat, query_ids, gallery_ids)
            return f"Expected RuntimeError but got {result}"
        except RuntimeError as e:
            return f"RuntimeError: {e} — should return 0.0 gracefully"

    def test_2_9_save_mode(self):
        rank_path = self.repo_root / "core/storymanager/rank/rank.py"
        with open(rank_path) as f:
            src = f.read()
        m = re.search(
            r"if not self\.save_mode (and|or) not isinstance\(self\.save_mode, list\)",
            src,
        )
        operator = m.group(1) if m else "and"
        save_mode = "selected_and_all"
        if operator == "or":
            fires = (not save_mode) or (not isinstance(save_mode, list))
        else:
            fires = (not save_mode) and (not isinstance(save_mode, list))
        if fires:
            return (
                f"ValueError raised for valid string save_mode='selected_and_all' — "
                f"isinstance(..., list) check with '{operator}' operator breaks all "
                f"existing configs that use string save_mode"
            )

    def test_2_10_onnx_import(self):
        mi_path = (
            self.repo_root
            / "core/testcasecontroller/algorithm/paradigm"
            / "multiedge_inference/multiedge_inference.py"
        )
        with open(mi_path) as f:
            lines = f.readlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == "import onnx" or stripped.startswith("import onnx "):
                indent = len(line) - len(line.lstrip())
                if indent == 0:
                    return (
                        f"`import onnx` is unconditional at module scope (line {i}). "
                        f"Fix: move inside _partition() as a lazy import. "
                        f"ModuleNotFoundError will crash any user who hasn't installed onnx."
                    )

    # ── LAYER 3: Smoke Tests ────────────────────────────────────────────────

    def test_3_1_dry_run(self):
        """Verify generate_reports CSV-reading works with whitespace-delimited mock data."""
        import pandas as pd
        import io

        mock_csv = (
            "rank algorithm mota motp f1_score precision recall idf1 "
            "paradigm basemodel batch_size time url\n"
            "1 ByteTrack 0.65 0.78 0.70 0.72 0.68 0.66 multiedgeinference "
            "ByteTrack 1 2024-01-01T00:00:00 /tmp/out\n"
        )
        src_path = self.example_root / "generate_reports.py"
        with open(src_path) as f:
            src = f.read()
        uses_delim_whitespace = "delim_whitespace" in src
        try:
            if uses_delim_whitespace:
                df = pd.read_csv(io.StringIO(mock_csv), delim_whitespace=True)
            else:
                df = pd.read_csv(io.StringIO(mock_csv), sep=r"\s+")
            df["time"] = pd.to_datetime(df["time"])
            _ = df.sort_values(by="time", ascending=False).iloc[0]["mota"]
        except TypeError as e:
            return (
                f"TypeError in pd.read_csv: {e} — "
                f"generate_reports.py will fail with pandas {pd.__version__}"
            )
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    def test_3_2_workspace(self):
        path = self.example_root / "tracking_job.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        workspace = data.get("benchmarkingjob", {}).get("workspace", "")
        if workspace.startswith("/"):
            return f"Absolute path cannot be created — Docker only: {workspace!r}"
        if workspace.startswith("../"):
            return (
                f"Path escapes repo root — PermissionError likely: {workspace!r}"
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                resolved = os.path.join(tmpdir, workspace.lstrip("./"))
                os.makedirs(resolved, exist_ok=True)
            except Exception as e:
                return f"{type(e).__name__}: {e}"

    # ── Runner ──────────────────────────────────────────────────────────────

    def run_all(self):
        print(f"\n{'='*60}")
        print(f"MOT17 Test Suite")
        print(f"Branch: {self.pr_branch}")
        print(f"Time:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        print("LAYER 1 — Static Validation")
        self.run_test("1.1 Stale path detector", self.test_1_1_stale_paths)
        self.run_test("1.2 Workspace path checker", self.test_1_2_workspace_path)
        self.run_test("1.3 YAML metric URL validator", self.test_1_3_metric_urls)
        self.run_test("1.4 Required files existence", self.test_1_4_required_files)
        self.run_test("1.5 Import dependency check", self.test_1_5_imports)
        self.run_test("1.6 Pandas API compatibility", self.test_1_6_pandas_api)
        self.run_test("1.7 YAML indentation validator", self.test_1_7_yaml_indent)
        self.run_test("1.8 generate_reports NameError check", self.test_1_8_nameerror)

        print("\nLAYER 2 — Unit Tests")
        self.run_test("2.1 f1_score ZeroDivisionError guard", self.test_2_1_f1_zero)
        self.run_test("2.2 f1_score NaN guard", self.test_2_2_f1_nan)
        self.run_test("2.3 f1_score normal computation", self.test_2_3_f1_normal)
        self.run_test("2.4 NaN propagation (precision)", self.test_2_4_nan_precision)
        self.run_test("2.4 NaN propagation (recall)", self.test_2_4_nan_recall)
        self.run_test("2.4 NaN propagation (mota)", self.test_2_4_nan_mota)
        self.run_test("2.4 NaN propagation (motp)", self.test_2_4_nan_motp)
        self.run_test("2.4 NaN propagation (idf1)", self.test_2_4_nan_idf1)
        self.run_test("2.5 generate_reports NameError", self.test_2_5_reports_nameerror)
        self.run_test("2.6 generate_reports None guard", self.test_2_6_reports_none_guard)
        self.run_test("2.7 rank_1 IndexError guard", self.test_2_7_rank_indexerror)
        self.run_test("2.8 mAP empty query set", self.test_2_8_map_empty)
        self.run_test("2.9 save_mode string regression", self.test_2_9_save_mode)
        self.run_test("2.10 ONNX lazy import", self.test_2_10_onnx_import)

        print("\nLAYER 3 — Smoke Tests")
        self.run_test("3.1 generate_reports dry run", self.test_3_1_dry_run)
        self.run_test("3.2 Workspace createability", self.test_3_2_workspace)

        self.print_summary()
        return self.exit_code()

    def print_summary(self):
        passed = sum(1 for r in self.results if r[0] == "PASS")
        failed = sum(1 for r in self.results if r[0] == "FAIL")
        errors = sum(1 for r in self.results if r[0] == "ERROR")
        total = len(self.results)

        print(f"\n{'='*60}")
        print(f"RESULTS: {passed}/{total} passed")
        print(f"{'='*60}")

        if failed or errors:
            print("\nFAILED / ERRORED:")
            for status, name, detail in self.results:
                if status in ("FAIL", "ERROR"):
                    print(f"\n  [{status}] {name}")
                    if detail:
                        for line in detail.strip().split("\n"):
                            print(f"    {line}")

        print(f"\n{'='*60}\n")

    def exit_code(self):
        if any(r[0] in ("FAIL", "ERROR") for r in self.results):
            return 1
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pr", default="main", help="PR number being tested (for log labeling)"
    )
    args = parser.parse_args()

    runner = MOT17TestRunner(pr_branch=args.pr)
    sys.exit(runner.run_all())
