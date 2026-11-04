# Vendored Sedna wheels

Ianvs runs its paradigms on [KubeEdge Sedna](https://github.com/kubeedge/sedna), installed from a
wheel kept here rather than from PyPI.

| Wheel | Used by |
|---|---|
| `sedna-0.6.0.1-py3-none-any.whl` | Ianvs core, CI and most examples |
| `sedna-0.6.0.3-py3-none-any.whl` | Lifelong-learning examples. Installed by the example's own setup script, for example `examples/cityscapes-synthia/lifelong_learning_bench/curb-detection/scripts/01_install_requirements.sh` |

**Install one Sedna wheel per environment, by name.** `pip install resources/third_party/*`
fails: pip cannot install two versions of `sedna` at once.

## Why a patched wheel

Lifelong learning in Ianvs uses Sedna's `LifelongLearning` job directly, in a single process,
with no KubeEdge cluster, no knowledge-base server and no edge node. Sedna 0.6 was written for
the cluster case, and eight of its defects show up only, or mainly, when it runs standalone like
this. They were found while restoring Ianvs's lifelong-learning examples (Phase IV of the example
restoration, [#1000](https://github.com/kubeedge/ianvs/pull/1000)) and are filed on Sedna as
[kubeedge/sedna#495](https://github.com/kubeedge/sedna/issues/495) to
[#502](https://github.com/kubeedge/sedna/issues/502).

SIG AI prefers fixing Sedna problems in the copy of Sedna that Ianvs vendors, so they
are fixed here. Ianvs core also works around five of them on its side (Sedna #495, #498, #499,
#500 and #502, in `core/testcasecontroller/algorithm/paradigm/sedna_lifelong_learning.py`), so core
still runs on `0.6.0.1`; there, results still vary between runs (Sedna #497) and every knowledge-base
call still retries for 25–35 s (Sedna #501).

## How 0.6.0.3 is built

`0.6.0.3` = `0.6.0.1` + [`patches/sedna-0.6.0.2.patch`](patches/sedna-0.6.0.2.patch) +
[`patches/sedna-0.6.0.3.patch`](patches/sedna-0.6.0.3.patch).

- `sedna-0.6.0.2.patch` is the `0.6.0.2` wheel proposed in
  [#488](https://github.com/kubeedge/ianvs/pull/488) (cityscapes-synthia semantic segmentation),
  which is not merged yet. It is carried unchanged.
- `sedna-0.6.0.3.patch` holds the Phase IV changes on top of it.

To rebuild the wheel, or to check that the vendored one contains exactly these patches:

```shell
pip install wheel
resources/third_party/build_sedna_wheel.sh OUTPUT_DIR   # writes sedna-0.6.0.3-py3-none-any.whl
resources/third_party/build_sedna_wheel.sh --check      # compares with the vendored wheel
```

### Changes in 0.6.0.2 (from #488)

Paths are relative to the `sedna` package.

| File | Change |
|---|---|
| `algorithms/seen_task_learning/task_update_decision/task_update_decision_finetune.py` | Defines and registers `UpdateStrategyDefault`, the default `task_update_decision`, which retrains each existing task model on the new samples ([#495](https://github.com/kubeedge/sedna/issues/495)) |
| `algorithms/seen_task_learning/__init__.py` | Imports `task_update_decision`, so its strategies are registered |
| `core/lifelong_learning/lifelong_learning.py` | `update()` adds `DATA_PATH_PREFIX` only to a relative `CLOUD_KB_INDEX` ([#499](https://github.com/kubeedge/sedna/issues/499)) |
| `core/lifelong_learning/lifelong_learning.py` | `train()` returns the knowledge-base index again, as in 0.4.x, instead of the training results (part of [#500](https://github.com/kubeedge/sedna/issues/500)) |
| `core/lifelong_learning/lifelong_learning.py` | Adds `inference_2()`, `my_evaluate()` and `my_inference()`, the methods of Ianvs's retired custom `sedna-0.4.1` |
| `core/lifelong_learning/lifelong_learning.py` | "KB update Fail" is logged at debug level when there is no knowledge-base server |
| `service/client.py` | `KBClient` returns at once when no knowledge-base server is set, instead of retrying each call for 25–35 s ([#501](https://github.com/kubeedge/sedna/issues/501)); `LCClient.send()` is skipped when the local controller is unset or on localhost |
| `core/base.py` | A local-controller connection error is logged at debug level |
| `common/constant.py` | `EDGE_KB_DIR` is `~/.sedna/kb` instead of `/var/lib/sedna/kb`, which needs root ([#502](https://github.com/kubeedge/sedna/issues/502)) |
| `algorithms/seen_task_learning/seen_task_learning.py` | Accepts task definition and allocation as objects as well as `{"method", "param"}` dicts; leaves out tasks without a model when loading an index |
| `algorithms/seen_task_learning/task_remodeling/task_remodeling.py` | No `IndexError` when there are no models |
| `core/lifelong_learning/knowledge_management/cloud_knowledge_management.py` | Model filtering accepts nested score dicts |
| `datasources/__init__.py` | `TxtDataParse` and `JsonlDataParse` keep samples as `dtype=object` arrays |

### Changes in 0.6.0.3

| File | Change |
|---|---|
| `algorithms/seen_task_learning/task_update_decision/task_update_decision_finetune.py` | `UpdateStrategyByFinetune` works without an `attribute` parameter, falling back to `("real", "sim")` as intended ([#496](https://github.com/kubeedge/sedna/issues/496)) |
| `algorithms/unseen_task_detection/unseen_sample_recognition/unseen_sample_recognition.py` | The default recognizer `SampleRegonitionDefault` no longer flips a coin for a single sample. It marks the sample seen or unseen by a stable hash of its file names, so a run gives the same result every time and on every machine. It also accepts keyword arguments such as `mode` ([#497](https://github.com/kubeedge/sedna/issues/497)) |
| `core/lifelong_learning/knowledge_management/cloud_knowledge_management.py` | The local copy of the knowledge-base index is kept in a temporary directory per job, not as `index.pkl` in the current directory, where a later job could reuse it ([#498](https://github.com/kubeedge/sedna/issues/498)) |
| `core/lifelong_learning/lifelong_learning.py` | Adds `inference_lifelong()`, lifelong inference from the cloud knowledge base without starting the edge inference service, and the `cloud_task_index` property, the index left by `train()` or `evaluate()` (rest of [#500](https://github.com/kubeedge/sedna/issues/500)). Same behaviour as the methods of the same name in Ianvs's `IanvsLifelongLearning`; additions only |

## What changes for existing users

Moving an environment from `0.6.0.1` to `0.6.0.3` changes these behaviours:

- `train()` returns the knowledge-base index, not the training results.
- The default unseen-sample recognizer gives repeatable results, so `samples_transfer_ratio`
  no longer varies between runs of the same configuration. It is still a placeholder that does
  not look at the knowledge base; an example that needs real detection configures its own.
- Without `edge_output_url`, Sedna creates its edge knowledge base under `~/.sedna/kb`.
- No `index.pkl` is written to the current directory.

The tests in `tests/core/test_sedna_wheel.py` check each fix and run only when `0.6.0.3` is
installed. `tests/core/test_sedna_lifelong_learning.py` runs the same `inference_lifelong()`
tests against Ianvs's subclass and, when present, the wheel's `LifelongLearning`.
