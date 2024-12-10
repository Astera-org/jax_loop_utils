# Copyright 2024 The CLU Authors.
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

"""Tests for the MemoryWriter."""

import pytest

from jax_loop_utils.metric_writers.memory_writer import MemoryWriter


def test_write_scalars():
    writer = MemoryWriter()
    assert writer.scalars == {}
    writer.write_scalars(0, {"a": 3, "b": 0.15})
    assert writer.scalars == {0: {"a": 3, "b": 0.15}}
    writer.write_scalars(2, {"b": 0.007})
    assert writer.scalars == {0: {"a": 3, "b": 0.15}, 2: {"b": 0.007}}


def test_write_scalars_fails_when_using_same_step():
    writer = MemoryWriter()
    writer.write_scalars(0, {})
    with pytest.raises(
        AssertionError, match=r"Key must be greater than the last inserted key\."
    ):
        writer.write_scalars(0, {})


def test_write_hparams():
    writer = MemoryWriter()
    assert writer.hparams is None
    writer.write_hparams({"a": 3, "b": 0.15})
    assert writer.hparams == {"a": 3, "b": 0.15}


def test_write_hparams_fails_when_called_more_than_once():
    writer = MemoryWriter()
    writer.write_hparams({})
    with pytest.raises(AssertionError, match=r"Hyperparameters can only be set once\."):
        writer.write_hparams({})
