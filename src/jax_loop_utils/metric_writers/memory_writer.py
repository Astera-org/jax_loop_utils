from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, OrderedDict, TypeVar

from .interface import Array, MetricWriter, Scalar

K = TypeVar("K")
V = TypeVar("V")


class _StrictlyOrderedDict(OrderedDict[K, V]):
    def __setitem__(self, key, value):
        if self:
            last_key = next(reversed(self))
            assert key > last_key, "Key must be greater than the last inserted key."
        super().__setitem__(key, value)


@dataclass
class MemoryWriterAudioEntry:
    audios: Mapping[str, Array]
    sample_rate: int


@dataclass
class MemoryWriterHistogramEntry:
    arrays: Mapping[str, Array]
    num_buckets: Optional[Mapping[str, int]]


class MemoryWriter(MetricWriter):
    scalars: OrderedDict[int, Mapping[str, Scalar]]
    images: OrderedDict[int, Mapping[str, Array]]
    videos: OrderedDict[int, Mapping[str, Array]]
    audios: OrderedDict[int, MemoryWriterAudioEntry]
    texts: OrderedDict[int, Mapping[str, str]]
    histograms: OrderedDict[int, MemoryWriterHistogramEntry]
    hparams: Optional[Mapping[str, object]]

    def __init__(self):
        self.scalars = _StrictlyOrderedDict()
        self.images = _StrictlyOrderedDict()
        self.videos = _StrictlyOrderedDict()
        self.audios = _StrictlyOrderedDict()
        self.texts = _StrictlyOrderedDict()
        self.histograms = _StrictlyOrderedDict()
        self.hparams = None

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        self.scalars[step] = scalars

    def write_images(self, step: int, images: Mapping[str, Array]):
        self.images[step] = images

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        self.videos[step] = videos

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        self.audios[step] = MemoryWriterAudioEntry(
            audios=audios, sample_rate=sample_rate
        )

    def write_texts(self, step: int, texts: Mapping[str, str]):
        self.texts[step] = texts

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Optional[Mapping[str, int]] = None,
    ):
        self.histograms[step] = MemoryWriterHistogramEntry(
            arrays=arrays, num_buckets=num_buckets
        )

    def write_hparams(self, hparams: Mapping[str, Any]):
        assert self.hparams is None, "Hyperparameters can only be set once."
        self.hparams = hparams

    def flush(self):
        pass

    def close(self):
        pass
