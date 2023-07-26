"""Defines metric classes for evaluating the performance of ECG reconstruction."""

import math
from abc import abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class AverageLoss:
    """The metric that tracks the average loss."""

    def __init__(self):
        self._weighted_loss_sum = 0.0
        self._batch_size_sum = 0

    def update(self, loss: float, batch_size: int) -> None:
        """Updates the metric with a new loss value."""
        self._weighted_loss_sum += loss * batch_size
        self._batch_size_sum += batch_size

    def update_and_log_step(
        self, loss: float, batch_size: int, writer: SummaryWriter, step: int
    ) -> None:
        """Updates the metric with a new weighted value, and logs this step to
        TensorBoard."""
        self.update(loss, batch_size)
        writer.add_scalar("Step/Loss", loss, step)

    def get_average(self) -> float:
        """Returns the average loss."""
        return self._weighted_loss_sum / self._batch_size_sum

    def log_epoch(self, writer: SummaryWriter, step: int) -> None:
        """Logs this epoch to TensorBoard."""
        epoch_loss = self.get_average()
        writer.add_scalar("Epoch/Loss", epoch_loss, step)

    def reset(self) -> None:
        """Resets the metric."""
        self._weighted_loss_sum = 0.0
        self._batch_size_sum = 0


class _LeadWiseMetricBase:
    """The base class for metrics that compute a scalar for each output lead."""

    def __init__(self, lead_names: Sequence[str]) -> None:
        self._lead_names = lead_names
        self._metric_batches: list[Tensor] = []

    def update(self, output: Tensor, target: Tensor) -> None:
        """Updates the metric with new batched outputs and targets."""
        metric_batch = self._compute(output.detach(), target.detach()).cpu()
        self._metric_batches.append(metric_batch)

    def update_and_log_step(
        self,
        output: Tensor,
        target: Tensor,
        writer: SummaryWriter,
        step: int,
    ) -> None:
        """Updates the metric with new batched outputs and targets, and logs this step
        to TensorBoard."""
        metric_batch = self._compute(output.detach(), target.detach()).cpu()
        self._metric_batches.append(metric_batch)
        lead_wise_metrics = metric_batch.mean(dim=0)
        writer.add_scalar(
            f"Step/{self._metric_name()}",
            self._postprocess(lead_wise_metrics.mean()),
            step,
        )
        for lead_name, value in zip(self._lead_names, lead_wise_metrics):
            writer.add_scalar(
                f"Step/{self._metric_name()}/{lead_name}",
                self._postprocess(value),
                step,
            )

    def get_average(self) -> float:
        """Returns the average metric value across all leads."""
        if len(self._metric_batches) > 1:
            self._metric_batches = [torch.cat(self._metric_batches)]
        return self._postprocess(self._metric_batches[0].mean())

    def log_epoch(self, writer: SummaryWriter, step: int) -> None:
        """Logs this epoch to TensorBoard."""
        if len(self._metric_batches) > 1:
            self._metric_batches = [torch.cat(self._metric_batches)]
        lead_wise_metrics = self._metric_batches[0].mean(dim=0)
        writer.add_scalar(
            f"Epoch/{self._metric_name()}",
            self._postprocess(lead_wise_metrics.mean()),
            step,
        )
        for lead_name, value in zip(self._lead_names, lead_wise_metrics):
            writer.add_scalar(
                f"Epoch/{self._metric_name()}/{lead_name}",
                self._postprocess(value),
                step,
            )

    def reset(self) -> None:
        """Resets the metric."""
        self._metric_batches.clear()

    def _postprocess(self, value: float | Tensor) -> float:
        if isinstance(value, Tensor):
            value = value.item()
        return value

    @abstractmethod
    def _metric_name(self) -> str:
        ...

    @abstractmethod
    def _compute(self, output: Tensor, target: Tensor) -> Tensor:
        ...


class RMSE(_LeadWiseMetricBase):
    """The metric that computes the root mean squared errors between the predicted and
    actual ECG signals."""

    def _postprocess(self, value: float | Tensor) -> float:
        return math.sqrt(super()._postprocess(value))

    def _metric_name(self) -> str:
        return "RMSE"

    def _compute(self, output: Tensor, target: Tensor) -> Tensor:
        return (output - target).square_().mean(dim=-1)


class PearsonR(_LeadWiseMetricBase):
    """The metric that computes the Pearson correlation coefficients between the
    predicted and actual ECG signals."""

    def _metric_name(self) -> str:
        return "PearsonR"

    def _compute(self, output: Tensor, target: Tensor) -> Tensor:
        def normalize(tensor: Tensor) -> Tensor:
            residual = tensor - tensor.mean(dim=-1, keepdim=True)
            norm = torch.linalg.norm(residual, dim=-1, keepdim=True)
            return residual.div_(norm)

        return normalize(output).mul_(normalize(target)).sum(dim=-1)


class ReconstructionMetrics:
    """An aggregate of metrics for evaluating the performance of ECG reconstruction."""

    def __init__(self, lead_names: Sequence[str], epoch: int = 1) -> None:
        """Initializes the metrics."""
        self.epoch = epoch
        self.average_loss = AverageLoss()
        self.rmse = RMSE(lead_names)
        self.pearson_r = PearsonR(lead_names)

    def update(self, loss: float, output: Tensor, target: Tensor) -> None:
        """Updates the metrics."""
        batch_size = output.size(dim=0)
        self.average_loss.update(loss, batch_size)
        self.rmse.update(output, target)
        self.pearson_r.update(output, target)

    def update_and_log_step(
        self,
        loss: float,
        output: Tensor,
        target: Tensor,
        writer: SummaryWriter,
        step: int,
    ) -> None:
        """Updates the metrics and logs this step to TensorBoard."""
        batch_size = output.size(dim=0)
        self.average_loss.update_and_log_step(loss, batch_size, writer, step)
        self.rmse.update_and_log_step(output, target, writer, step)
        self.pearson_r.update_and_log_step(output, target, writer, step)
        writer.add_scalar("Epoch", self.epoch, step)

    def log_epoch(self, writer: SummaryWriter, step: int) -> None:
        """Logs this epoch to TensorBoard."""
        self.average_loss.log_epoch(writer, step)
        self.rmse.log_epoch(writer, step)
        self.pearson_r.log_epoch(writer, step)

    def reset(self, epoch: int) -> None:
        """Resets the metrics."""
        self.epoch = epoch
        self.average_loss.reset()
        self.rmse.reset()
        self.pearson_r.reset()
