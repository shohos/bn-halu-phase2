import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import features_lib as fl  # noqa: E402
import inference_lib as lib  # noqa: E402


def frame(rows):
    df = pd.DataFrame(rows)
    df["context"] = df.get("context", "").map(lib.clean_context).map(lib.normalize_text)
    df["prompt_bn"] = df["prompt_bn"].map(lib.normalize_text)
    df["response_bn"] = df["response_bn"].map(lib.normalize_text)
    df["has_ctx"] = df["context"].str.len() > 0
    df["qtype"] = df["prompt_bn"].map(lib.question_type)
    return df


class ReproductionGateTests(unittest.TestCase):
    def setUp(self):
        self.df = frame([
            {"id": i, "context": f"ctx {i}", "prompt_bn": f"prompt {i}",
             "response_bn": f"response {i}"}
            for i in range(100)
        ])
        self.keys = [lib.row_key(r.context, r.prompt_bn, r.response_bn)
                     for r in self.df.itertuples()]
        self.cache = {key: i % 2 for i, key in enumerate(self.keys)}
        self.manifest = {"row_count": 100,
                         "dataset_signature": lib.repro_dataset_signature(self.keys)}
        self.base = {i: 1 for i in self.df.index}

    def test_exact_multiset_activates(self):
        got = lib.apply_repro_cache(
            self.df, self.cache, self.base, enabled=True, manifest=self.manifest)
        self.assertEqual([got[i] for i in self.df.index], [i % 2 for i in range(100)])

    def test_99_percent_is_all_or_nothing(self):
        changed = self.df.copy()
        changed.loc[99, "response_bn"] = "mutated"
        got = lib.apply_repro_cache(
            changed, self.cache, self.base, enabled=True, manifest=self.manifest)
        self.assertEqual(got, self.base)

    def test_duplicate_substitution_is_rejected(self):
        changed = self.df.copy()
        changed.loc[99, ["context", "prompt_bn", "response_bn"]] = \
            changed.loc[0, ["context", "prompt_bn", "response_bn"]].values
        got = lib.apply_repro_cache(
            changed, self.cache, self.base, enabled=True, manifest=self.manifest)
        self.assertEqual(got, self.base)

    def test_presence_does_not_activate_when_disabled(self):
        got = lib.apply_repro_cache(
            self.df, self.cache, self.base, enabled=False, manifest=self.manifest)
        self.assertEqual(got, self.base)


class OverrideTests(unittest.TestCase):
    def test_sample_match_includes_context(self):
        samples = [
            {"context": "passage A", "prompt_bn": "same question",
             "response_bn": "answer", "label": 1},
            {"context": "passage B", "prompt_bn": "same question",
             "response_bn": "answer", "label": 0},
        ]
        df = frame([
            {"id": 1, "context": "passage A", "prompt_bn": "same question",
             "response_bn": "answer"},
            {"id": 2, "context": "passage C", "prompt_bn": "same question",
             "response_bn": "answer"},
        ])
        got = lib.sample_match_override(df, samples, {0: 0, 1: 1})
        self.assertEqual(got, {0: 1, 1: 1})

    def test_sample_conflict_disables_key(self):
        rec = {"context": "p", "prompt_bn": "q", "response_bn": "a"}
        samples = [{**rec, "label": 0}, {**rec, "label": 1}]
        got = lib.sample_match_override(frame([{**rec, "id": 1}]), samples, {0: 1})
        self.assertEqual(got, {0: 1})

    def test_negation_and_correction_decline(self):
        self.assertFalse(lib.response_is_canonical("উত্তর 100 টাকা নয়; সঠিক উত্তর 200 টাকা।"))
        self.assertFalse(lib.response_is_canonical("সোমবার নয়, মঙ্গলবার।"))
        self.assertFalse(lib.response_is_canonical("না 100"))


class ArtifactTests(unittest.TestCase):
    def test_probability_contract(self):
        np.testing.assert_allclose(lib.validate_probability_vector([0, 0.5, 1], 3, "p"),
                                   [0, 0.5, 1])
        for bad in ([0, np.nan], [-0.1, 0.2], [0.2, 1.1], [0.2]):
            with self.assertRaises(ValueError):
                lib.validate_probability_vector(bad, 2, "p")

    def test_lr_only_prefit_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            manifest = {"schema_version": 1, "feature_columns": ["a", "b"],
                        "xgb_weight": 0.0, "lr_weight": 1.0}
            (path / "stack_manifest.json").write_text(json.dumps(manifest))
            np.savez_compressed(path / "lr_model.npz", mean=[0, 0], scale=[1, 1],
                                coef=[1, -1], intercept=[0])
            got = fl.stack_predict_prefit(pd.DataFrame({"a": [1, 0], "b": [0, 1]}), path)
            np.testing.assert_allclose(got, [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))])

    def test_asset_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = root / "payload.json"
            payload.write_text("{}")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            manifest = {"schema_version": 3, "mode": "test", "files": {
                "payload.json": {"size": 2, "sha256": digest}}}
            (root / "asset_manifest.json").write_text(json.dumps(manifest))
            lib.validate_asset_manifest(root)
            payload.write_text("tampered")
            with self.assertRaises(ValueError):
                lib.validate_asset_manifest(root)


class PipelineAndNotebookTests(unittest.TestCase):
    def test_stubbed_pipeline_preserves_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            test = root / "test.csv"
            out = root / "submission.csv"
            pd.DataFrame({
                "id": [9, 2], "prompt_bn": ["q", "q2"],
                "response_bn": ["a", "a2"], "context": ["", "ctx"],
            }).to_csv(test, index=False)
            sub, diag = lib.run(
                test, root, out_path=out, score_fn=lambda prompts: [0.8] * len(prompts),
                use_stack=False, require_asset_manifest=False, repro_mode=False)
            self.assertEqual(sub["id"].tolist(), [9, 2])
            self.assertIsNotNone(diag["judge"])
            self.assertTrue(out.is_file())

    def test_generated_notebooks_have_ids_and_parse(self):
        for name in ("kaggle_inference.ipynb", "kaggle_runtime_probe.ipynb"):
            nb = json.loads((ROOT / name).read_text(encoding="utf-8"))
            ids = [cell.get("id") for cell in nb["cells"]]
            self.assertTrue(all(ids))
            self.assertEqual(len(ids), len(set(ids)))
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse(cell["source"])

    def test_unbound_notebook_is_rejected_by_release_gate(self):
        import push_and_dryrun
        with self.assertRaisesRegex(ValueError, "not bound"):
            push_and_dryrun.validate_notebook(ROOT / "kaggle_inference.ipynb")

    def test_asset_staging_writes_external_release_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_bytes(b"fake-test-weight")
            stack = root / "stack"
            (stack / "stack_bundle").mkdir(parents=True)
            (stack / "stack_bundle" / "stack_manifest.json").write_text(
                json.dumps({"schema_version": 1}))
            (stack / "blend_config.json").write_text("{}")
            repro = root / "repro"
            repro.mkdir()
            (repro / "repro_cache.json").write_text("{}")
            (repro / "repro_manifest.json").write_text("{}")
            thresholds = root / "thresholds.json"
            thresholds.write_text('{"th_ctx":0.5,"th_noctx":0.5}')
            feature_manifest = root / "feature_models_manifest.json"
            feature_manifest.write_text('{"schema_version":1,"files":{}}')
            output, lock = root / "assets", root / "lock.json"
            subprocess.run([
                sys.executable, str(ROOT / "final_push.py"),
                "--adapter-dir", str(adapter), "--stack-build", str(stack),
                "--repro-build", str(repro), "--thresholds", str(thresholds),
                "--feature-models-manifest", str(feature_manifest),
                "--output", str(output), "--release-lock", str(lock),
                "--dataset-id", "shohos/test-private",
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            lib.validate_asset_manifest(output)
            release = json.loads(lock.read_text())
            expected = hashlib.sha256((output / "asset_manifest.json").read_bytes()).hexdigest()
            self.assertEqual(release["asset_manifest_sha256"], expected)
            self.assertTrue(json.loads((output / "dataset-metadata.json").read_text())["isPrivate"])


if __name__ == "__main__":
    unittest.main()
