"""ECG reconstruction models with our Fastformer architecture and models with a mix of
Fastformer and U-Net architectures."""

import math
from collections.abc import Sequence
from typing import Any

import torch
from optuna import Trial
from torch import nn, Tensor
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
)

from ..reconstructor import Reconstructor
from ..unet.unet import DoubleConv, DownSample, OutConv, UpSample

_DEFAULT_BERT_CONFIG = {
    "hidden_size": 256,
    "hidden_dropout_prob": 0.2,
    "num_hidden_layers": 2,
    "hidden_act": "gelu",
    "num_attention_heads": 16,
    "intermediate_size": 256,
    "max_position_embeddings": 8192,
    "type_vocab_size": 2,
    "vocab_size": 100000,
    "layer_norm_eps": 1e-12,
    "initializer_range": 0.02,
    "pooler_type": "weightpooler",
    "enable_fp16": False,
}


class _FastSelfAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the "
                f"number of attention heads ({config.num_attention_heads})"
            )
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.query_attention = nn.Linear(config.hidden_size, config.num_attention_heads)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_attention = nn.Linear(config.hidden_size, config.num_attention_heads)
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def _transpose_for_scores(self, tensor: Tensor) -> Tensor:
        new_shape = tensor.size()[:-1] + (
            self.config.num_attention_heads,
            self.attention_head_size,
        )
        return tensor.view(new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Args:
            hidden_states: A tensor of shape `(batch_size, seq_len, hidden_size)`.
            attention_mask: A tensor of shape `(batch_size, 1, seq_len)`.

        Returns:
            A tensor of shape `(batch_size, seq_len, hidden_size)`.
        """
        _, seq_len, _ = hidden_states.size()
        # (batch_size, seq_len, hidden_size)
        mixed_query_layer = self.query(hidden_states)
        # (batch_size, seq_len, hidden_size)
        mixed_key_layer = self.key(hidden_states)
        # (batch_size, num_attention_heads, seq_len)
        query_for_score = self.query_attention(mixed_query_layer).transpose(
            1, 2
        ) / math.sqrt(self.attention_head_size)
        query_for_score += attention_mask
        # (batch_size, num_attention_heads, 1, seq_len)
        query_weight = self.softmax(query_for_score).unsqueeze(2)
        # (batch_size, num_attention_heads, seq_len, hidden_size / num_attention_heads)
        query_layer = self._transpose_for_scores(mixed_query_layer)
        # (batch_size, 1, hidden_size)
        pooled_query = (
            torch.matmul(query_weight, query_layer).transpose(1, 2).flatten(start_dim=2)
        )
        # (batch_size, seq_len, hidden_size)
        pooled_query_repeat = pooled_query.repeat(1, seq_len, 1)
        # (batch_size, seq_len, hidden_size)
        mixed_query_key_layer = mixed_key_layer * pooled_query_repeat
        # (batch_size, num_attention_heads, seq_len)
        query_key_score = (
            self.key_attention(mixed_query_key_layer)
            / math.sqrt(self.attention_head_size)
        ).transpose(1, 2)
        query_key_score += attention_mask
        # (batch_size, num_attention_heads, 1, seq_len)
        query_key_weight = self.softmax(query_key_score).unsqueeze(dim=2)
        # (batch_size, num_attention_heads, seq_len, hidden_size / num_attention_heads)
        key_layer = self._transpose_for_scores(mixed_query_key_layer)
        # (batch_size, num_attention_heads, 1, hidden_size / num_attention_heads)
        pooled_key = torch.matmul(query_key_weight, key_layer)
        # (batch_size, seq_len, num_attention_heads, hidden_size / num_attention_heads)
        weighted_value = (pooled_key * query_layer).transpose(1, 2)
        # (batch_size, seq_len, hidden_size)
        weighted_value = weighted_value.flatten(start_dim=2)
        # (batch_size, seq_len, hidden_size)
        return self.transform(weighted_value) + mixed_query_layer


class _FastAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.self_attention = _FastSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Args:
            hidden_states: A tensor of shape `(batch_size, seq_len, hidden_size)`.
            attention_mask: A tensor of shape `(batch_size, 1, seq_len)`.

        Returns:
            A tensor of shape `(batch_size, seq_len, hidden_size)`.
        """
        # (batch_size, seq_len, hidden_size)
        self_output = self.self_attention(hidden_states, attention_mask)
        # (batch_size, seq_len, hidden_size)
        return self.output(self_output, hidden_states)


class _FastformerLayer(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = _FastAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Args:
            hidden_states: A tensor of shape `(batch_size, seq_len, hidden_size)`.
            attention_mask: A tensor of shape `(batch_size, 1, seq_len)`.

        Returns:
            A tensor of shape `(batch_size, seq_len, hidden_size)`.
        """
        # (batch_size, seq_len, hidden_size)
        attention_output = self.attention(hidden_states, attention_mask)
        # (batch_size, seq_len, intermediate_size)
        intermediate_output = self.intermediate(attention_output)
        # (batch_size, seq_len, hidden_size)
        return self.output(intermediate_output, attention_output)


class _FastformerEncoder(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleList(
            [_FastformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, input_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Args:
            input_embeddings: A tensor of shape `(batch_size, seq_len, hidden_size)`.
            attention_mask: A tensor of shape `(batch_size, seq_len)`.

        Returns:
            A tensor of shape `(batch_size, seq_len, hidden_size)`.
        """
        # (batch_size, 1, seq_len)
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.type(
            next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        batch_size, seq_len, _ = input_embeddings.size()
        # (seq_len)
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=input_embeddings.device
        )
        # (batch_size, seq_len)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        # (batch_size, seq_len, hidden_size)
        position_embeddings = self.position_embeddings(position_ids)
        # (batch_size, seq_len, hidden_size)
        embeddings = input_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        # (batch_size, seq_len, hidden_size)
        hidden_states = self.dropout(embeddings)
        for layer in self.encoders:
            hidden_states = layer(hidden_states, extended_attention_mask)
        return hidden_states


class Fastformer(Reconstructor):
    """A transformer-based ECG reconstruction model aimed for fast computation."""

    max_batch_size = 16

    def __init__(
        self,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        bert_config_dict: dict[str, Any],
    ) -> None:
        """
        Initializes the model.

        Args:
            in_leads: Indices of the input leads.
            out_leads: Indices of the output leads.
            bert_config_dict: A dictionary of parameters to `BertConfig`.
        """
        super().__init__()
        bert_config = BertConfig.from_dict(bert_config_dict)
        self.signal_embedding = nn.Linear(len(in_leads), bert_config.hidden_size)
        self.fastformer_model = _FastformerEncoder(bert_config)
        # self.out_conv = OutConv(bert_config.hidden_size, len(out_leads))
        self.out_conv = nn.Conv1d(bert_config.hidden_size, len(out_leads), kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: A tensor of shape `(batch_size, in_channels, seq_len)`.

        Returns:
            A tensor of shape `(batch_size, out_channels, seq_len)`.
        """
        # (batch_size, seq_len, in_channels)
        inputs = inputs.transpose(-2, -1)
        # (batch_size, seq_len)
        mask = torch.ones(inputs.size()[:2], device=inputs.device)
        # (batch_size, seq_len, hidden_size)
        embeddings = self.signal_embedding(inputs)
        # (batch_size, seq_len, hidden_size)
        signal_vec = self.fastformer_model(embeddings, mask)
        # (batch_size, hidden_size, seq_len)
        signal_vec = signal_vec.transpose(-2, -1)
        # (batch_size, out_channels, seq_len)
        return self.out_conv(signal_vec)

    @staticmethod
    def default_config() -> dict:
        return {
            "reconstructor": {
                "args": {
                    "bert_config_dict": {
                        **_DEFAULT_BERT_CONFIG.copy(),
                        # This should be equal to the maximum number of samples of ECG
                        # signals.
                        "max_position_embeddings": 5000,
                    },
                }
            },
            "optimizer": {
                "type": "AdamW",
                "args": {
                    "lr": 2e-3,
                    "weight_decay": 5e-2,
                },
            },
            "lr_scheduler": {
                "type": "ReduceLROnPlateau",
                "args": {
                    "factor": 0.5,
                    "patience": 5,
                },
            },
            "dataloader": {
                "common": {"batch_size": 128},
            },
        }

    @staticmethod
    def suggest_config(trial: Trial) -> dict:
        config = Fastformer.default_config()

        bert_cfg_dict = _DEFAULT_BERT_CONFIG.copy()
        model_size = trial.suggest_int("model_size", 64, 256, step=32)
        bert_cfg_dict.update(dict(
            hidden_size = model_size,
            intermediate_size = model_size,
            hidden_act = trial.suggest_categorical('hidden_act', ['gelu', 'silu']),
            num_attention_heads = trial.suggest_categorical("num_attention_heads", [1, 2, 4, 8]), 
            num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 4),
        ))
        config["reconstructor"]["args"].update(
            bert_config_dict=bert_cfg_dict
        )

        return config
    
class FastformerPlus(Reconstructor):
    """A transformer-based ECG reconstruction model aimed for fast computation."""

    max_batch_size = 16

    def __init__(
        self,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        bert_config_dict: dict[str, Any],
    ) -> None:
        """
        Initializes the model.

        Args:
            in_leads: Indices of the input leads.
            out_leads: Indices of the output leads.
            bert_config_dict: A dictionary of parameters to `BertConfig`.
        """
        super().__init__()
        bert_config = BertConfig.from_dict(bert_config_dict)
        # self.signal_embedding = nn.Linear(len(in_leads), bert_config.hidden_size)
        self.signal_embedding = nn.Sequential(
            # nn.Conv1d(len(in_leads), 3 * len(in_leads), kernel_size=3, padding='same', dilation=1),
            # nn.Conv1d(3 * len(in_leads), bert_config.hidden_size//2, kernel_size=3, padding='same', dilation=2),
            DoubleConv(len(in_leads), bert_config.hidden_size//2),
            DoubleConv(bert_config.hidden_size//2, bert_config.hidden_size, dilation=4),

        )
        self.fastformer_model = _FastformerEncoder(bert_config)
        # self.out_conv = OutConv(bert_config.hidden_size, len(out_leads))
        # self.out_conv = nn.Conv1d(bert_config.hidden_size, len(out_leads), kernel_size=1)
        self.out_conv = nn.Sequential(
            DoubleConv(bert_config.hidden_size, bert_config.hidden_size),
            nn.Conv1d(bert_config.hidden_size, len(out_leads), kernel_size=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: A tensor of shape `(batch_size, in_channels, seq_len)`.

        Returns:
            A tensor of shape `(batch_size, out_channels, seq_len)`.
        """
        inputs = self.signal_embedding(inputs)
        # (batch_size, seq_len, in_channels)
        inputs = inputs.transpose(-2, -1)
        # (batch_size, seq_len)
        mask = torch.ones(inputs.size()[:2], device=inputs.device)
        # (batch_size, seq_len, hidden_size)
        signal_vec = self.fastformer_model(inputs, mask)
        # (batch_size, hidden_size, seq_len)
        signal_vec = signal_vec.transpose(-2, -1)
        # (batch_size, out_channels, seq_len)
        return self.out_conv(signal_vec)

    @staticmethod
    def default_config() -> dict:
        return {
            "reconstructor": {
                "args": {
                    "bert_config_dict": {
                        **_DEFAULT_BERT_CONFIG.copy(),
                        # This should be equal to the maximum number of samples of ECG
                        # signals.
                        "max_position_embeddings": 5000,
                    },
                }
            },
            "optimizer": {
                "type": "AdamW",
                "args": {
                    "lr": 2e-3,
                    "weight_decay": 5e-2,
                },
            },
            "lr_scheduler": {
                "type": "ReduceLROnPlateau",
                "args": {
                    "factor": 0.5,
                    "patience": 5,
                },
            },
            "dataloader": {
                "common": {"batch_size": 128},
            },
        }

    @staticmethod
    def suggest_config(trial: Trial) -> dict:
        config = Fastformer.default_config()

        bert_cfg_dict = _DEFAULT_BERT_CONFIG.copy()
        model_size = trial.suggest_int("model_size", 32, 128, step=32)
        bert_cfg_dict.update(dict(
            hidden_size = model_size,
            intermediate_size = model_size,
            hidden_act = trial.suggest_categorical('hidden_act', ['gelu', 'silu']),
            num_attention_heads = trial.suggest_categorical("num_attention_heads", [1, 2, 4, 8]), 
            num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 4),
        ))
        config["reconstructor"]["args"].update(
            bert_config_dict=bert_cfg_dict
        )

        return config


class UFastformer(Reconstructor):
    """An ECG reconstruction model based on the U-Net, but with an additional
    `Fastformer` between the encoder and decoder."""
    max_batch_size = 8
    def __init__(
        self,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        bert_config_dict: dict[str, Any],
        min_channels: int = 16,
        min_dilation: int = 1,
        dilation_rate: int = 2,
        bilinear: bool = True,
        unet_num_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.in_layer = DoubleConv(len(in_leads), min_channels, **kwargs)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(unet_num_layers):
            in_channels = min_channels << i
            out_channels = in_channels * 2
            dilation = min_dilation * dilation_rate ** (i + 1)
            self.encoder.append(
                DownSample(in_channels, out_channels, dilation=dilation, **kwargs)
            )
            self.decoder.append(
                UpSample(out_channels, in_channels, bilinear, dilation=1, **kwargs)
            )
        self.decoder = self.decoder[::-1]
        self.out_layer = OutConv(min_channels, len(out_leads))

        center_in_channels = out_channels
        bert_config = BertConfig.from_dict(bert_config_dict)
        self.input_embedding = nn.Linear(center_in_channels, bert_config.hidden_size)
        self.fastformer_model = _FastformerEncoder(bert_config)
        self.output_transform = OutConv(bert_config.hidden_size, center_in_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: A tensor of shape `(batch_size, in_channels, seq_len)`.

        Returns:
            A tensor of shape `(batch_size, out_channels, seq_len)`.
        """
        # (batch_size, min_channels, seq_len)
        encoder_outputs = [self.in_layer(inputs)]
        for layer in self.encoder:
            encoder_outputs.append(layer(encoder_outputs[-1]))
        # (batch_size, seq_len, min_channels << unet_num_layers)
        transformer_inputs = encoder_outputs.pop().transpose(-2, -1)
        # (batch_size, seq_len)
        mask = torch.ones(
            transformer_inputs.size()[:2], device=transformer_inputs.device
        )
        # (batch_size, seq_len, hidden_size)
        src = self.input_embedding(transformer_inputs)
        # (batch_size, seq_len, hidden_size)
        outputs = self.fastformer_model(src, mask)
        # (batch_size, min_channels << unet_num_layers, seq_len)
        direct_inputs = self.output_transform(outputs.transpose(-2, -1))
        for layer in self.decoder:
            skip_inputs = encoder_outputs.pop()
            direct_inputs = layer(direct_inputs, skip_inputs)
        # (batch_size, out_channels, seq_len)
        return self.out_layer(direct_inputs)

    @staticmethod
    def default_config() -> dict:
        return {
            "reconstructor": {
                "args": {
                    "bert_config_dict": {
                        **_DEFAULT_BERT_CONFIG.copy(),
                        # Assume maximum number of samples to be 5000 and minimum number
                        # of down-sampling layers to be 2.
                        "max_position_embeddings": 1250,
                    },
                    "min_channels": 16,
                    "min_dilation": 1,
                    "dilation_rate": 2,
                    "bilinear": True,
                    "unet_num_layers": 2,
                }
            },
            "optimizer": {
                "type": "AdamW",
                "args": {
                    "lr": 2e-3,
                    "weight_decay": 5e-2,
                },
            },
            "lr_scheduler": {
                "type": "ReduceLROnPlateau",
                "args": {
                    "factor": 0.5,
                    "patience": 5,
                },
            },
            "dataloader": {
                "common": {"batch_size": 128},
            },
        }

    @staticmethod
    def suggest_config(trial: Trial) -> dict:
        config = UFastformer.default_config()
        bert_cfg_dict = _DEFAULT_BERT_CONFIG.copy()
        model_size = trial.suggest_int("model_size", 64, 256, step=32)
        bert_cfg_dict.update(dict(
            hidden_size = model_size,
            intermediate_size = model_size,
            hidden_act = trial.suggest_categorical('hidden_act', ['gelu', 'silu']),
            num_attention_heads = trial.suggest_categorical("num_attention_heads", [1, 2, 4, 8]), 
            num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 4),
        ))
        config["reconstructor"]["args"].update(
            bert_config_dict=bert_cfg_dict,
            unet_num_layers=trial.suggest_int("unet_num_layers", 3, 5),
            min_channels=trial.suggest_int("min_channels", 16, 32, step=4),
            kernel_size=trial.suggest_int("kernel_size", 3, 33, step=2),
            min_dilation=trial.suggest_int("min_dilation", 1, 4),
            dilation_rate=trial.suggest_int("dilation_rate", 1, 8),
        )

        return config

class FastformerZero(Reconstructor):
    """A transformer-based ECG reconstruction model aimed for fast computation."""

    max_batch_size = 16

    def __init__(
        self,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        bert_config_dict: dict[str, Any],
    ) -> None:
        """
        Initializes the model.

        Args:
            in_leads: Indices of the input leads.
            out_leads: Indices of the output leads.
            bert_config_dict: A dictionary of parameters to `BertConfig`.
        """
        super().__init__()
        bert_config = BertConfig.from_dict(bert_config_dict)
        # self.signal_embedding = nn.Linear(len(in_leads), bert_config.hidden_size)
        self.signal_embedding = nn.Sequential(
            # nn.Conv1d(len(in_leads), 3 * len(in_leads), kernel_size=3, padding='same', dilation=1),
            # nn.Conv1d(3 * len(in_leads), bert_config.hidden_size//2, kernel_size=3, padding='same', dilation=2),
            DoubleConv(len(in_leads), bert_config.hidden_size//2),
            DoubleConv(bert_config.hidden_size//2, bert_config.hidden_size, dilation=4),

        )
        self.fastformer_model = nn.Conv1d(bert_config.hidden_size, bert_config.hidden_size, kernel_size=1)
        self.out_conv = nn.Sequential(
            DoubleConv(bert_config.hidden_size, bert_config.hidden_size),
            nn.Conv1d(bert_config.hidden_size, len(out_leads), kernel_size=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: A tensor of shape `(batch_size, in_channels, seq_len)`.

        Returns:
            A tensor of shape `(batch_size, out_channels, seq_len)`.
        """
        inputs = self.signal_embedding(inputs)
        # (batch_size, seq_len, in_channels)
        signal_vec = self.fastformer_model(inputs)
        # (batch_size, out_channels, seq_len)
        return self.out_conv(signal_vec)

    @staticmethod
    def default_config() -> dict:
        return {
            "reconstructor": {
                "args": {
                    "bert_config_dict": {
                        **_DEFAULT_BERT_CONFIG.copy(),
                        # This should be equal to the maximum number of samples of ECG
                        # signals.
                        "max_position_embeddings": 5000,
                    },
                }
            },
            "optimizer": {
                "type": "AdamW",
                "args": {
                    "lr": 2e-3,
                    "weight_decay": 5e-2,
                },
            },
            "lr_scheduler": {
                "type": "ReduceLROnPlateau",
                "args": {
                    "factor": 0.5,
                    "patience": 5,
                },
            },
            "dataloader": {
                "common": {"batch_size": 128},
            },
        }

    @staticmethod
    def suggest_config(trial: Trial) -> dict:
        config = Fastformer.default_config()

        bert_cfg_dict = _DEFAULT_BERT_CONFIG.copy()
        model_size = trial.suggest_int("model_size", 32, 128, step=32)
        bert_cfg_dict.update(dict(
            hidden_size = model_size,
            intermediate_size = model_size,
            hidden_act = trial.suggest_categorical('hidden_act', ['gelu', 'silu']),
            num_attention_heads = trial.suggest_categorical("num_attention_heads", [1, 2, 4, 8]), 
            num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 4),
        ))
        config["reconstructor"]["args"].update(
            bert_config_dict=bert_cfg_dict
        )

        return config


class FastformerStuff(Reconstructor):
    """A transformer-based ECG reconstruction model aimed for fast computation."""

    max_batch_size = 16

    def __init__(
        self,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        bert_config_dict: dict[str, Any],
    ) -> None:
        """
        Initializes the model.

        Args:
            in_leads: Indices of the input leads.
            out_leads: Indices of the output leads.
            bert_config_dict: A dictionary of parameters to `BertConfig`.
        """
        super().__init__()
        bert_config = BertConfig.from_dict(bert_config_dict)
        # self.signal_embedding = nn.Linear(len(in_leads), bert_config.hidden_size)
        self.signal_embedding = nn.Sequential(
            # nn.Conv1d(len(in_leads), 3 * len(in_leads), kernel_size=3, padding='same', dilation=1),
            # nn.Conv1d(3 * len(in_leads), bert_config.hidden_size//2, kernel_size=3, padding='same', dilation=2),
            DoubleConv(len(in_leads), bert_config.hidden_size//2),
            DoubleConv(bert_config.hidden_size//2, bert_config.hidden_size, dilation=4),

        )
        self.fastformer_model = nn.Sequential(
            DoubleConv(bert_config.hidden_size, bert_config.hidden_size, dilation=8),
            DoubleConv(bert_config.hidden_size, bert_config.hidden_size, dilation=16),
            DoubleConv(bert_config.hidden_size, bert_config.hidden_size, dilation=32),
            DoubleConv(bert_config.hidden_size, bert_config.hidden_size, dilation=64),

        )
        self.out_conv = nn.Sequential(
            DoubleConv(bert_config.hidden_size, bert_config.hidden_size),
            nn.Conv1d(bert_config.hidden_size, len(out_leads), kernel_size=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: A tensor of shape `(batch_size, in_channels, seq_len)`.

        Returns:
            A tensor of shape `(batch_size, out_channels, seq_len)`.
        """
        inputs = self.signal_embedding(inputs)
        # (batch_size, seq_len, in_channels)
        signal_vec = self.fastformer_model(inputs)
        # (batch_size, out_channels, seq_len)
        return self.out_conv(signal_vec)

    @staticmethod
    def default_config() -> dict:
        return {
            "reconstructor": {
                "args": {
                    "bert_config_dict": {
                        **_DEFAULT_BERT_CONFIG.copy(),
                        # This should be equal to the maximum number of samples of ECG
                        # signals.
                        "max_position_embeddings": 5000,
                    },
                }
            },
            "optimizer": {
                "type": "AdamW",
                "args": {
                    "lr": 2e-3,
                    "weight_decay": 5e-2,
                },
            },
            "lr_scheduler": {
                "type": "ReduceLROnPlateau",
                "args": {
                    "factor": 0.5,
                    "patience": 5,
                },
            },
            "dataloader": {
                "common": {"batch_size": 128},
            },
        }

    @staticmethod
    def suggest_config(trial: Trial) -> dict:
        config = Fastformer.default_config()

        bert_cfg_dict = _DEFAULT_BERT_CONFIG.copy()
        model_size = trial.suggest_int("model_size", 32, 128, step=32)
        bert_cfg_dict.update(dict(
            hidden_size = model_size,
            intermediate_size = model_size,
            hidden_act = trial.suggest_categorical('hidden_act', ['gelu', 'silu']),
            num_attention_heads = trial.suggest_categorical("num_attention_heads", [1, 2, 4, 8]), 
            num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 4),
        ))
        config["reconstructor"]["args"].update(
            bert_config_dict=bert_cfg_dict
        )

        return config