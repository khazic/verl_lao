# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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


def _patch_output_gate_slicing():
    """Fix gate not being sliced when num_query_groups < tp_size.

    Without this patch, _apply_output_gate crashes with a shape mismatch
    because the gate tensor has more elements than the attention output.

    Ref: https://github.com/NVIDIA/Megatron-LM/pull/3529
    """
    from megatron.core import parallel_state
    from megatron.core.transformer.attention import SelfAttention

    _original_get_qkv = SelfAttention.get_query_key_value_tensors

    def _patched_get_qkv(self, *args, **kwargs):
        result = _original_get_qkv(self, *args, **kwargs)
        output_gate = args[2] if len(args) > 2 else kwargs.get("output_gate", False)
        if output_gate and isinstance(result, tuple) and len(result) == 4:
            query, key, value, gate = result
            num_query_groups = getattr(self.config, "num_query_groups", None)
            if num_query_groups is not None and num_query_groups < self.world_size:
                tp_rank = parallel_state.get_tensor_model_parallel_rank()
                groups_per_rank = self.world_size // num_query_groups
                idx = tp_rank % groups_per_rank
                size = self.num_attention_heads_per_partition // groups_per_rank
                gate = gate[:, :, idx * size : (idx + 1) * size, :]
            return query, key, value, gate
        return result

    SelfAttention.get_query_key_value_tensors = _patched_get_qkv


def apply_patch_mbridge():
    try:
        from megatron.core.utils import \
            get_tensor_model_parallel_group_if_none  # noqa: F401
    except ImportError:
        import warnings

        import megatron.core.utils
        import torch
        from megatron.core import parallel_state

        def get_tensor_model_parallel_group_if_none(tp_group, is_expert=False, check_initialized=True):
            """Issue a deprecation warning if tp_group is None and return the default tp group."""
            if not torch.distributed.is_initialized():
                return None
            if tp_group is None:
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    warnings.warn(
                        "Warning: tp_group is None, using default tp group. Passing tp_group will be mandatory soon",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if is_expert:
                    tp_group = parallel_state.get_expert_tensor_parallel_group(check_initialized=check_initialized)
                else:
                    tp_group = parallel_state.get_tensor_model_parallel_group(check_initialized=check_initialized)
            return tp_group

        megatron.core.utils.get_tensor_model_parallel_group_if_none = get_tensor_model_parallel_group_if_none

    _patch_output_gate_slicing()
