# SPDX-License-Identifier: Apache-2.0
"""Select Qwen3Next/Qwen3.5 MTP weights before materializing tensors."""

import json
import os
import time
from collections.abc import Generator

import torch
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from vllm.config import ModelConfig
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import get_model_architecture
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP
from vllm.model_executor.models.qwen3_next_mtp import Qwen3NextMTP

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class Qwen3NextMTPLoader(DefaultModelLoader):
    """Default loader with MTP selection before opening shards/get_tensor."""

    def _needs_weight(self, name: str) -> bool:
        # Both MTP load_weights implementations accept these checkpoint names;
        # Qwen3.5's language_model prefix is remapped by its model loader.
        return (
            name.startswith("mtp.") or "embed_tokens" in name or "lm_head" in name
        ) and not weight_utils.should_skip_weight(name, self.local_expert_ids)

    def get_all_weights(
        self, model_config: ModelConfig, model: torch.nn.Module
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        # Validate the full index first: selecting MTP-only patterns would
        # make upstream reject the omitted main-model shards as missing.
        hf_folder, files, _ = self._prepare_weights(
            model_config.model,
            None,
            model_config.revision,
            False,
            ["*.safetensors"],
        )
        with open(os.path.join(hf_folder, SAFE_WEIGHTS_INDEX_NAME)) as f:
            weight_map = json.load(f)["weight_map"]
        needed_files = {
            filename
            for name, filename in weight_map.items()
            if self._needs_weight(name)
        }
        selected_files = [
            path for path in files if os.path.relpath(path, hf_folder) in needed_files
        ]
        logger.info(
            "Qwen MTP weight filter: loading %d/%d safetensors shards",
            len(selected_files),
            len(files),
        )
        self.counter_before_loading_weights = time.perf_counter()
        for path in weight_utils.tqdm(
            sorted(selected_files, key=weight_utils._natural_sort_key),
            desc="Loading Qwen MTP checkpoint shards",
            disable=not weight_utils.enable_tqdm(self.load_config.use_tqdm_on_load),
        ):
            with weight_utils.safe_open(path, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if self._needs_weight(name):
                        yield name, f.get_tensor(name)


def maybe_use_qwen3_next_mtp_loader(
    loader: BaseModelLoader, model_config: ModelConfig
) -> BaseModelLoader:
    """Use selective mmap reads for indexed local Qwen3Next/Qwen3.5 MTP.

    The unset strategy also takes this lazy path, avoiding automatic whole-file
    prefetch for the draft. Explicit eager/prefetch strategies stay upstream.
    """
    if type(loader) is not DefaultModelLoader:
        return loader
    load_config = loader.load_config
    if (
        load_config.load_format not in ("auto", "safetensors")
        or load_config.safetensors_load_strategy not in (None, "lazy")
        or load_config.model_loader_extra_config.get("enable_multithread_load", False)
        or not os.path.isfile(os.path.join(model_config.model, SAFE_WEIGHTS_INDEX_NAME))
    ):
        return loader
    model_cls, _ = get_model_architecture(model_config)
    if not issubclass(model_cls, (Qwen3NextMTP, Qwen3_5MoeMTP)):
        return loader
    return Qwen3NextMTPLoader(load_config)
