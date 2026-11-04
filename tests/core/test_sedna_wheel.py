# Copyright 2026 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Sedna fixes in the patched sedna-0.6.0.3 wheel
(resources/third_party/README.md). Skipped when another Sedna is installed."""

import importlib.metadata
import os
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    importlib.metadata.version("sedna") != "0.6.0.3",
    reason="needs the patched sedna-0.6.0.3 wheel")

# pylint: disable=wrong-import-position
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.common.constant import KBResourceConstant
from sedna.core.lifelong_learning import LifelongLearning  # registers its algorithms
from sedna.datasources import BaseDataSource


def _sample(*names):
    data = BaseDataSource(data_type="test")
    data.x = [list(names)]
    return data


def _is_unseen(recognizer, data, **kwargs):
    _, unseen = recognizer(data, **kwargs)
    return len(unseen.x) > 0


def test_update_strategy_default_comes_from_sedna():
    # kubeedge/sedna#495
    cls = ClassFactory.get_cls(ClassType.KM, "UpdateStrategyDefault")
    assert cls.__module__.startswith("sedna.")


def test_update_strategy_by_finetune_works_without_attribute():
    # kubeedge/sedna#496
    cls = ClassFactory.get_cls(ClassType.KM, "UpdateStrategyByFinetune")
    assert tuple(cls({}).attribute) == ("real", "sim")
    assert cls({}, attribute="a, b").attribute == ["a", "b"]


def test_default_recognizer_is_repeatable():
    # kubeedge/sedna#497
    recognizer = ClassFactory.get_cls(ClassType.UTD, "SampleRegonitionDefault")({})
    samples = [_sample(f"/data/img{i}.png", f"/data/depth{i}.png") for i in range(40)]

    first = [_is_unseen(recognizer, s) for s in samples]
    second = [_is_unseen(recognizer, s) for s in samples]

    assert first == second
    assert any(first) and not all(first)  # still a seen/unseen split


def test_default_recognizer_ignores_dataset_location_and_accepts_kwargs():
    # kubeedge/sedna#497
    recognizer = ClassFactory.get_cls(ClassType.UTD, "SampleRegonitionDefault")({})
    for i in range(10):
        here = _sample(f"/a/img{i}.png", f"/a/depth{i}.png")
        there = _sample(f"/b/c/img{i}.png", f"/b/c/depth{i}.png")
        assert _is_unseen(recognizer, here) == _is_unseen(recognizer, there, mode="x")


def test_local_index_is_per_job_and_not_in_the_current_directory(monkeypatch):
    # kubeedge/sedna#498
    from sedna.core.lifelong_learning.knowledge_management import (
        base_knowledge_management, cloud_knowledge_management)
    monkeypatch.setattr(base_knowledge_management.BaseKnowledgeManagement,
                        "__init__", lambda self, *args: None)
    config = {"task_index": "index.pkl"}

    first = cloud_knowledge_management.CloudKnowledgeManagement(config, None, None)
    second = cloud_knowledge_management.CloudKnowledgeManagement(config, None, None)

    assert os.path.isabs(first.local_task_index_url)
    assert os.path.dirname(first.local_task_index_url) != os.getcwd()
    assert first.local_task_index_url != second.local_task_index_url


def test_update_keeps_an_absolute_index_path(tmp_path, monkeypatch):
    # kubeedge/sedna#499
    from sedna.common.file_ops import FileOps
    index = str(tmp_path / "index.pkl")
    downloads = []

    def fake_download(src, dst):
        downloads.append(src)
        raise StopIteration

    monkeypatch.setenv("CLOUD_KB_INDEX", index)
    monkeypatch.setattr(FileOps, "download", fake_download)
    job = LifelongLearning.__new__(LifelongLearning)
    job.cloud_knowledge_management = SimpleNamespace(
        task_index="unused", local_task_index_url=str(tmp_path / "local.pkl"))

    with pytest.raises(StopIteration):
        job.update("train-data")

    assert downloads == [index]


def test_kb_client_without_server_returns_at_once(tmp_path):
    # kubeedge/sedna#501
    from sedna.service.client import KBClient
    client = KBClient(kbserver=None)
    index = tmp_path / "index.pkl"
    index.write_bytes(b"")

    start = time.monotonic()
    assert client.upload_file(str(index)) == str(index)
    assert client.update_db(str(index)) == str(index)
    assert client.update_task_status("[]") is None
    assert time.monotonic() - start < 1


def test_edge_kb_dir_does_not_need_root():
    # kubeedge/sedna#502
    assert not KBResourceConstant.EDGE_KB_DIR.value.startswith("/var/lib")
