"""MLflow implementation of MetricWriter interface."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from absl import logging

import mlflow
import mlflow.config
from jax_loop_utils.metric_writers.interface import (
    Array,
    Scalar,
)
from jax_loop_utils.metric_writers.interface import (
    MetricWriter as MetricWriterInterface,
)


class MetricWriter(MetricWriterInterface):
    """MLflow implementation of MetricWriter."""

    def __init__(self, experiment_name: str | None = None):
        """Initialize MLflow writer.

        Args:
            experiment_name: Name of the MLflow experiment. If None, uses active experiment.
        """
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        self._active_run = mlflow.active_run()
        if not self._active_run:
            self._active_run = mlflow.start_run()

    def write_summaries(
        self,
        step: int,
        values: Mapping[str, Array],
        metadata: Mapping[str, Any] | None = None,
    ):
        """MLflow doesn't support generic summaries, so we log as metrics."""
        for key, value in values.items():
            mlflow.log_metric(key, float(np.array(value).mean()), step=step)

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        """Write scalar metrics to MLflow."""
        floats = {key: float(np.array(value)) for key, value in scalars.items()}
        mlflow.log_metrics(floats, step=step)

    def write_images(self, step: int, images: Mapping[str, Array]):
        """Write images to MLflow."""
        for key, image_array in images.items():
            mlflow.log_image(image_array, key=key, step=step)

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        """MLflow doesn't support video logging directly."""
        # this could be supported if we convert the video to a file
        # and log the file as an artifact.
        logging.log_first_n(
            logging.WARNING,
            "mlflow.MetricWriter does not support writing videos.",
            1,
        )

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        """MLflow doesn't support audio logging directly."""
        # this could be supported if we convert the video to a file
        # and log the file as an artifact.
        logging.log_first_n(
            logging.WARNING,
            "mlflow.MetricWriter does not support writing audios.",
            1,
        )

    def write_texts(self, step: int, texts: Mapping[str, str]):
        """Write text artifacts to MLflow."""
        for key, text in texts.items():
            mlflow.log_text(text, f"{key}_step_{step}.txt")

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Mapping[str, int] | None = None,
    ):
        """MLflow doesn't support histogram logging directly.

        https://github.com/mlflow/mlflow/issues/8145
        """
        logging.log_first_n(
            logging.WARNING,
            "mlflow.MetricWriter does not support writing histograms.",
            1,
        )

    def write_hparams(self, hparams: Mapping[str, Any]):
        """Log hyperparameters to MLflow."""
        if not isinstance(hparams, dict):
            hparams = dict(hparams)
        mlflow.log_params(hparams)

    def flush(self):
        """Flushes all logged data."""
        mlflow.flush_artifact_async_logging()
        mlflow.flush_async_logging()
        mlflow.flush_trace_async_logging()

    def close(self):
        """End the MLflow run."""
        if self._active_run:
            mlflow.end_run()
        self.flush()
