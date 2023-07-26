"""Defines the `Trainer` class and its configuration `TrainerConfig`, for training and
evaluating ECG reconstruction models."""

import datetime
import logging
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

import numpy as np
import optuna
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from . import reconstructor as reconstructor_module
from .dataset import Dataset, DatasetConfig
from .metric import ReconstructionMetrics
from .reconstructor.reconstructor import Reconstructor
from .util.device import get_device
from .util.lead import get_lead_names
from .util.path import get_project_root_dir
from .util.progress import ProgressCounter

_logger = logging.getLogger(__name__)


class _TypeAndArgs(TypedDict):
    type: str | type
    args: Mapping[str, Any]


class TrainerConfig(TypedDict):
    """Configuration for creating a `Trainer`."""

    in_leads: Sequence[int]
    """Indices of the input leads."""

    out_leads: Sequence[int]
    """Indices of the output leads."""

    max_epochs: int
    """The maximum number of epochs to train for."""

    accumulate_grad_batches: int
    """The number of batches to accumulate gradients over in training."""

    dataset: Mapping[Literal["common", "train", "eval"], DatasetConfig]
    """Configuration for creating the datasets used for training and evaluation."""

    dataloader: Mapping[Literal["common", "train", "eval"], Mapping[str, Any]]
    """Arguments to the dataloaders used for training and evaluation."""

    reconstructor: _TypeAndArgs
    """The type and arguments for the reconstruction model."""

    optimizer: Optional[_TypeAndArgs]
    """The type and arguments for the optimizer."""

    lr_scheduler: Optional[_TypeAndArgs]
    """The type and arguments for the learning rate scheduler."""


class Trainer:
    """The utility class for training and evaluating an ECG reconstruction model."""

    def __init__(self, config: TrainerConfig) -> None:
        _logger.debug("Create trainer from config: %s", config)
        self.config = config
        self.train_dataset: Dataset = None
        self.eval_dataset: Dataset = None
        self._create_datasets()
        self.train_dataloader: DataLoader = None
        self.eval_dataloader: DataLoader = None
        self._create_dataloaders()
        self.reconstructor: Reconstructor = None
        self._create_reconstructor()
        lead_names = get_lead_names()
        self.optimizer = None
        self._create_optimizer()
        self.lr_scheduler = None
        self._create_lr_scheduler()
        _logger.debug("Create ReconstructionMetrics")
        self.metrics = ReconstructionMetrics(
            tuple(lead_names[i] for i in config["out_leads"])
        )

    def save_checkpoint(self, filename: str) -> None:
        """Saves the state dictionaries of the reconstruction model, the optimizer (if
        any), and the learning rate scheduler (if any) to a checkpoint file."""
        _logger.debug("Save checkpoint to: %s", filename)
        checkpoint = {"reconstructor": self.reconstructor.state_dict()}
        if self.optimizer is not None:
            checkpoint["optimizer"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str) -> None:
        """Loads the state dictionaries of the reconstruction model, the optimizer (if
        any), and the learning rate scheduler (if any) from a checkpoint file."""
        _logger.debug("Load checkpoint from: %s", filename)
        checkpoint = torch.load(filename, map_location=get_device())

        if "reconstructor" not in checkpoint:
            _logger.warning("Checkpoint does not have 'reconstructor'")
        else:
            self.reconstructor.load_state_dict(checkpoint["reconstructor"], strict=False)

        if self.optimizer is None:
            if "optimizer" in checkpoint:
                _logger.warning("Checkpoint has 'optimizer' but trainer does not")
        else:
            if "optimizer" not in checkpoint:
                _logger.warning("Checkpoint does not have 'optimizer'")
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.lr_scheduler is None:
            if "lr_scheduler" in checkpoint:
                _logger.warning("Checkpoint has 'lr_scheduler' but trainer does not")
        else:
            if "lr_scheduler" not in checkpoint:
                _logger.warning("Checkpoint does not have 'lr_scheduler'")
            else:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def fit(
        self,
        name: Optional[str] = None,
        tensorboard: bool = True,
        checkpoint: bool = True,
        log_every_n_steps: int = 1,
        trial: optuna.Trial = None,
    ) -> float:
        """
        Runs the training and evaluation loop.

        Args:
            name: If given, the name of the reconstruction model used to create log
              directories. If not given, the type name of the model is used.
            tensorboard: Whether to log to TensorBoard.
            checkpoint: Whether to save checkpoints.
            log_every_n_steps: The number of steps between logging. Set to a
              non-positive number to disable logging steps.
        Return:
            The average loss of the last epoch's validation
        """
        _logger.debug("Fit the reconstruction model")
        device = get_device()
        global_step = -1
        min_eval_loss = float("inf")
        if trial:
            _logger.info(
                "During hyperparameter tuning, checkpointing and Tensorboard logging "
                "are always turned off"
            )
            tensorboard = None
            checkpoint = None
        tensorboard_dir, checkpoint_dir = self._create_dirs(
            name, tensorboard, checkpoint
        )
        if not tensorboard:
            train_writer = None
            eval_writer = None
        else:
            _logger.debug("Create TensorBoard writers")
            train_writer = SummaryWriter(tensorboard_dir / "Train")
            eval_writer = SummaryWriter(tensorboard_dir / "Evaluation")

        for epoch in range(1, self.config["max_epochs"] + 1):
            _logger.info("Epoch %d", epoch)

            self.reconstructor.train()
            self.optimizer.zero_grad()
            self.metrics.reset(epoch)
            step_in_epoch = 0
            counter = ProgressCounter(self.train_dataloader, prefix="Train")
            for batch in counter:
                # if step_in_epoch > 3806:
                #     print(batch["input"].shape)
                output = self.reconstructor(batch["input"].to(device))
                target = batch["target"].to(device)
                loss = nn.functional.mse_loss(output, target)
                loss.backward()
                loss = loss.item()
                if (
                    is_optimizer_step := (step_in_epoch + 1)
                    % self.config["accumulate_grad_batches"]
                    == 0 or step_in_epoch == len(self.train_dataloader) - 1
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if isinstance(self.lr_scheduler, optim.lr_scheduler.OneCycleLR):
                        self.lr_scheduler.step()
                    global_step += 1
                if (
                    is_optimizer_step
                    and tensorboard
                    and log_every_n_steps > 0
                    and (global_step + 1) % log_every_n_steps == 0
                ):
                    self.metrics.update_and_log_step(
                        loss, output, target, train_writer, global_step
                    )
                else:
                    self.metrics.update(loss, output, target)
                counter.postfix = {
                    "batch_loss": loss,
                    "average_loss": self.metrics.average_loss.get_average(),
                }
                step_in_epoch += 1

            if tensorboard:
                self.metrics.log_epoch(train_writer, global_step)
            _logger.info(
                "Loss=%.4g, RMSE=%.4g, Pearson-R=%.4g",
                self.metrics.average_loss.get_average(),
                self.metrics.rmse.get_average(),
                self.metrics.pearson_r.get_average(),
            )

            self._eval_impl(epoch, progress_prefix="Evaluation")
            average_loss = self.metrics.average_loss.get_average()
            _logger.info(
                "Loss=%.4g, RMSE=%.4g, PearsonR=%.4g",
                average_loss,
                self.metrics.rmse.get_average(),
                self.metrics.pearson_r.get_average(),
            )

            if trial is not None:
                if np.isnan(average_loss):
                    _logger.warning("Average loss over the past epoch is NaN")
                trial.report(
                    average_loss, epoch
                )  # just report a value to be minimized on
                if trial.should_prune():
                    _logger.info("Current trial has been pruned")
                    raise optuna.TrialPruned()

            if tensorboard:
                self.metrics.log_epoch(eval_writer, global_step)

            if checkpoint:
                checkpoint_name = f"epoch={epoch}_loss={average_loss:.4g}.pth"
                self.save_checkpoint(checkpoint_dir / checkpoint_name)
                _logger.debug("Create 'latest' tag for the checkpoint")
                (checkpoint_dir / "latest").write_text(checkpoint_name)

            if average_loss < min_eval_loss:
                min_eval_loss = average_loss
                _logger.info("Best evaluation loss so far")
                if checkpoint:
                    _logger.debug("Create 'best' tag for the checkpoint")
                    (checkpoint_dir / "best").write_text(checkpoint_name)

        if tensorboard:
            _logger.debug("Close TensorBoard writer")
            train_writer.close()
            eval_writer.close()

        return average_loss
    
    def resume(
        self,
        checkpoint_dir: str,
        tensorboard_dir: str = None,
        log_every_n_steps: int = 1,
    ) -> float:
        """
        Runs the training and evaluation loop.

        Args:
            name: If given, the name of the reconstruction model used to create log
              directories. If not given, the type name of the model is used.
            tensorboard: Whether to log to TensorBoard.
            checkpoint: Whether to save checkpoints.
            log_every_n_steps: The number of steps between logging. Set to a
              non-positive number to disable logging steps.
        Return:
            The average loss of the last epoch's validation
        """
        _logger.debug("Fit the reconstruction model")
        device = get_device()
        global_step = -1

        checkpoint_dir = Path(checkpoint_dir)
        if tensorboard_dir:
            tensorboard_dir = Path(tensorboard_dir)
        
        import re
        with open(checkpoint_dir / "best") as f:
            min_eval_loss = float(re.findall(r"=([0-9.]+)\.pth", f.read())[0])
        _logger.info(f"Current min loss {min_eval_loss:.4g}")

        if not tensorboard_dir:
            train_writer = None
            eval_writer = None
        else:
            train_writer = SummaryWriter(tensorboard_dir / "Train")
            eval_writer = SummaryWriter(tensorboard_dir / "Evaluation")

        with open(checkpoint_dir / "latest") as f:
            lastest_checkpoint_name = f.read()
            current_epoch = int(re.findall(r"epoch=([0-9]+)", lastest_checkpoint_name)[0]) + 1

        _logger.info(f"Resume from {lastest_checkpoint_name}")
        
        self.load_checkpoint(checkpoint_dir / lastest_checkpoint_name)

        for epoch in range(current_epoch, self.config["max_epochs"] + 1):
            _logger.info("Epoch %d", epoch)

            self.reconstructor.train()
            self.optimizer.zero_grad()
            self.metrics.reset(epoch)
            step_in_epoch = 0
            counter = ProgressCounter(self.train_dataloader, prefix="Train")
            for batch in counter:
                # if step_in_epoch > 3806:
                #     print(batch["input"].shape)
                output = self.reconstructor(batch["input"].to(device))
                target = batch["target"].to(device)
                loss = nn.functional.mse_loss(output, target)
                loss.backward()
                loss = loss.item()
                if (
                    is_optimizer_step := (step_in_epoch + 1)
                    % self.config["accumulate_grad_batches"]
                    == 0 or step_in_epoch == len(self.train_dataloader) - 1
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if isinstance(self.lr_scheduler, optim.lr_scheduler.OneCycleLR):
                        self.lr_scheduler.step()
                    global_step += 1
                if (
                    is_optimizer_step
                    and tensorboard_dir
                    and log_every_n_steps > 0
                    and (global_step + 1) % log_every_n_steps == 0
                ):
                    self.metrics.update_and_log_step(
                        loss, output, target, train_writer, global_step
                    )
                else:
                    self.metrics.update(loss, output, target)
                counter.postfix = {
                    "batch_loss": loss,
                    "average_loss": self.metrics.average_loss.get_average(),
                }
                step_in_epoch += 1

            if tensorboard_dir:
                self.metrics.log_epoch(train_writer, global_step)
            _logger.info(
                "Loss=%.4g, RMSE=%.4g, Pearson-R=%.4g",
                self.metrics.average_loss.get_average(),
                self.metrics.rmse.get_average(),
                self.metrics.pearson_r.get_average(),
            )

            self._eval_impl(epoch, progress_prefix="Evaluation")
            average_loss = self.metrics.average_loss.get_average()
            _logger.info(
                "Loss=%.4g, RMSE=%.4g, PearsonR=%.4g",
                average_loss,
                self.metrics.rmse.get_average(),
                self.metrics.pearson_r.get_average(),
            )

            if tensorboard_dir:
                self.metrics.log_epoch(eval_writer, global_step)

            checkpoint_name = f"epoch={epoch}_loss={average_loss:.4g}.pth"
            self.save_checkpoint(checkpoint_dir / checkpoint_name)
            _logger.debug("Create 'latest' tag for the checkpoint")
            (checkpoint_dir / "latest").write_text(checkpoint_name)

            if average_loss < min_eval_loss:
                min_eval_loss = average_loss
                _logger.info("Best evaluation loss so far")
                _logger.debug("Create 'best' tag for the checkpoint")
                (checkpoint_dir / "best").write_text(checkpoint_name)

        if tensorboard_dir:
            _logger.debug("Close TensorBoard writer")
            train_writer.close()
            eval_writer.close()


    def test(self) -> None:
        """Tests the model on the evaluation dataset for one epoch."""
        _logger.debug("Test the reconstruction model")
        self._eval_impl(epoch=1, progress_prefix="Test")

    def _eval_impl(self, epoch: int, progress_prefix: str) -> None:
        device = get_device()
        self.reconstructor.eval()
        self.metrics.reset(epoch)
        counter = ProgressCounter(self.eval_dataloader, prefix=progress_prefix)
        with torch.no_grad():
            for batch in counter:
                output = self.reconstructor(batch["input"].to(device))
                target = batch["target"].to(device)
                loss = nn.functional.mse_loss(output, target)
                loss = loss.item()
                self.metrics.update(loss, output, target)
                counter.postfix = {
                    "batch_loss": loss,
                    "average_loss": self.metrics.average_loss.get_average(),
                }

    def _create_datasets(self) -> None:
        common_dataset_config = self.config["dataset"].get("common", {})
        _logger.debug("Create the training dataset")
        self.train_dataset = Dataset(
            {
                "in_leads": self.config["in_leads"],
                "out_leads": self.config["out_leads"],
                **common_dataset_config,
                **self.config["dataset"].get("train", {}),
            }
        )
        _logger.debug("Create the evaluation dataset")
        self.eval_dataset = Dataset(
            {
                "in_leads": self.config["in_leads"],
                "out_leads": self.config["out_leads"],
                **common_dataset_config,
                **self.config["dataset"].get("eval", {}),
            }
        )

    def _create_dataloaders(self) -> None:
        base_dataloader_config = {"persistent_workers": True}
        device = get_device()
        if device.type in ("cuda", "xpu"):
            # The pin_memory_device accepts only a string.
            # Issue: https://github.com/pytorch/pytorch/issues/82583
            device_str = device.type
            if device.index is not None:
                device_str += f":{device.index}"
            base_dataloader_config.update(pin_memory=True, pin_memory_device=device_str)
        common_dataloader_config = self.config["dataloader"].get("common", {})

        def update_batch_size_for_grad_accum(args: dict[str, Any], name: str) -> None:
            batch_size = args["batch_size"]
            accumulate_grad_batches = self.config["accumulate_grad_batches"]
            if batch_size % accumulate_grad_batches != 0:
                _logger.warning(
                    "Batch size for %s dataloader is changed from %d to %d, because it "
                    "is not divisible by the number of batches %d for gradient "
                    "accumulation",
                    name,
                    batch_size,
                    batch_size // accumulate_grad_batches * accumulate_grad_batches,
                    accumulate_grad_batches,
                )
            args["batch_size"] = batch_size // accumulate_grad_batches

        _logger.debug("Create the training dataloader")
        train_dataloader_args = {
            "shuffle": True,
            **base_dataloader_config,
            **common_dataloader_config,
            **self.config["dataloader"].get("train", {}),
        }
        update_batch_size_for_grad_accum(train_dataloader_args, "training")
        self.train_dataloader = DataLoader(self.train_dataset, **train_dataloader_args)

        _logger.debug("Create the evaluation dataloader")
        eval_dataloader_args = {
            "shuffle": False,
            **base_dataloader_config,
            **common_dataloader_config,
            **self.config["dataloader"].get("eval", {}),
        }
        update_batch_size_for_grad_accum(eval_dataloader_args, "evaluation")
        self.eval_dataloader = DataLoader(self.eval_dataset, **eval_dataloader_args)

    def _create_reconstructor(self) -> None:
        reconstructor_type = Reconstructor.resolve_type(
            self.config["reconstructor"]["type"]
        )
        _logger.debug("Create the reconstruction model")
        self.reconstructor = reconstructor_type(
            **{
                "in_leads": self.config["in_leads"],
                "out_leads": self.config["out_leads"],
                **self.config["reconstructor"]["args"],
            },
        )
        device = get_device()
        _logger.debug("Move the reconstruction model to %s", device)
        self.reconstructor.to(device)
        if self.config["optimizer"] is None and device.type == "xpu":
            _logger.debug("Use ipex to optimize the model for evaluation")
            try:
                import intel_extension_for_pytorch as ipex  # pylint: disable=import-outside-toplevel
            except ImportError as error:
                raise RuntimeError(
                    "Expect ipex to be installed when using XPU"
                ) from error

            self.reconstructor.eval()
            self.reconstructor = ipex.optimize(self.reconstructor)

    def _create_optimizer(self) -> None:
        if self.config["optimizer"] is None:
            _logger.debug("No optimizer is specified")
            return
        _logger.debug("Create the optimizer")
        optimizer_type = self.config["optimizer"]["type"]
        if isinstance(optimizer_type, str):
            optimizer_type = getattr(optim, optimizer_type)
        self.optimizer = optimizer_type(
            self.reconstructor.parameters(), **self.config["optimizer"]["args"]
        )
        if get_device().type == "xpu":
            _logger.debug("Use ipex to optimize the model for training")
            try:
                import intel_extension_for_pytorch as ipex  # pylint: disable=import-outside-toplevel
            except ImportError as error:
                raise RuntimeError(
                    "Expect ipex to be installed when using XPU"
                ) from error

            self.reconstructor.train()
            self.reconstructor, self.optimizer = ipex.optimize(
                self.reconstructor, optimizer=self.optimizer
            )

    def _create_lr_scheduler(self) -> None:
        if self.config["lr_scheduler"] is None:
            _logger.debug("No learning rate scheduler is specified")
            return
        _logger.debug("Create the learning rate scheduler")

        def make_lr_scheduler(type_and_args: _TypeAndArgs) -> Any:
            scheduler_type = type_and_args["type"]
            if isinstance(scheduler_type, str):
                scheduler_type = getattr(optim.lr_scheduler, scheduler_type)
            args = type_and_args["args"].copy()
            if "lr_lambda" in args and isinstance(args["lr_lambda"], str):
                args["lr_lambda"] = eval(args["lr_lambda"])  # pylint: disable=eval-used
            if scheduler_type is optim.lr_scheduler.OneCycleLR:
                args = {
                    "epochs": self.config["max_epochs"],
                    "steps_per_epoch": len(self.train_dataloader)
                    // self.config["accumulate_grad_batches"],
                    **args,
                }
            if scheduler_type is optim.lr_scheduler.ChainedScheduler:
                assert set(args) == {"schedulers"}
                return scheduler_type(list(map(make_lr_scheduler, args["schedulers"])))
            return scheduler_type(self.optimizer, **args)

        self.lr_scheduler = make_lr_scheduler(self.config["lr_scheduler"])

    def _create_dirs(
        self, name: Optional[str], tensorboard: bool, checkpoint: bool
    ) -> tuple[Optional[Path], Optional[Path]]:
        if not tensorboard and not checkpoint:
            return None, None

        project_root_dir = get_project_root_dir()
        if name is None:
            name = Reconstructor.resolve_type(
                self.config["reconstructor"]["type"]
            ).__name__
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")

        if not tensorboard:
            tensorboard_base_dir = None
        else:
            tensorboard_base_dir = project_root_dir / "src" / "tensorboard" / name
            if not tensorboard_base_dir.exists():
                _logger.debug(
                    "Create TensorBoard base directory: %s", tensorboard_base_dir
                )
                tensorboard_base_dir.mkdir(parents=True)

        if not checkpoint:
            checkpoint_base_dir = None
        else:
            checkpoint_base_dir = project_root_dir / "src" / "checkpoints" / name
            if not checkpoint_base_dir.exists():
                _logger.debug(
                    "Create checkpoint base directory: %s", checkpoint_base_dir
                )
                checkpoint_base_dir.mkdir(parents=True)

        dir_name_pattern = re.compile(re.escape(timestamp) + r"(?:-v(?P<version>\d+))?")
        max_version = 0

        def update_max_version(item: Path) -> None:
            nonlocal max_version
            if match := dir_name_pattern.fullmatch(item.name):
                version_str = match.group("version")
                max_version = max(
                    max_version, 1 if version_str is None else int(version_str)
                )

        if tensorboard_base_dir is not None:
            for item in tensorboard_base_dir.iterdir():
                update_max_version(item)
        if checkpoint_base_dir is not None:
            for item in checkpoint_base_dir.iterdir():
                update_max_version(item)
        version = max_version + 1
        dir_name = f"{timestamp}-v{version}" if version > 1 else timestamp

        if tensorboard_base_dir is None:
            tensorboard_dir = None
        else:
            tensorboard_dir = tensorboard_base_dir / dir_name
            _logger.debug("Create TensorBoard directory: %s", tensorboard_dir)
            tensorboard_dir.mkdir()
            _logger.debug("Create 'latest' tag for the TensorBoard directory")
            (tensorboard_base_dir / "latest").write_text(dir_name)

        if checkpoint_base_dir is None:
            checkpoint_dir = None
        else:
            checkpoint_dir = checkpoint_base_dir / dir_name
            _logger.debug("Create checkpoint directory: %s", checkpoint_dir)
            checkpoint_dir.mkdir()
            _logger.debug("Create 'latest' tag for the checkpoint directory")
            (checkpoint_base_dir / "latest").write_text(dir_name)
            trainer_config_path = checkpoint_dir / "trainer_config.yaml"
            _logger.debug("Save trainer configuration to: %s", trainer_config_path)
            with trainer_config_path.open("w", encoding="utf-8") as config_file:
                yaml.dump(self.config, config_file, Dumper=yaml.Dumper)

        return tensorboard_dir, checkpoint_dir
