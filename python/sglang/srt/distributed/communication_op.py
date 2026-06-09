# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed

from sglang.srt.utils.custom_op import register_custom_op

from .parallel_state import (
    get_attn_tp_group,
    get_moe_ep_group,
    get_moe_tp_group,
    get_tp_group,
)


def _tensor_model_parallel_all_reduce_impl(input_: torch.Tensor) -> torch.Tensor:
    return get_tp_group().all_reduce(input_)


@register_custom_op(
    op_name="tbo_tensor_model_parallel_all_reduce",
    fake_impl=lambda input_: torch.empty_like(input_),
)
def _tbo_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return _tensor_model_parallel_all_reduce_impl(input_)


def _use_tbo_all_reduce_custom_op() -> bool:
    # Imported lazily to avoid a circular import: server_args is imported very
    # early (e.g. via sglang.srt.configs) and that chain pulls in this module.
    from sglang.srt.server_args import get_global_server_args

    try:
        server_args = get_global_server_args()
    except ValueError:
        return False
    return (
        server_args.enable_two_batch_overlap
        and server_args.tbo_all_reduce_as_custom_op
    )


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    if _use_tbo_all_reduce_custom_op():
        return _tbo_tensor_model_parallel_all_reduce(input_)
    return _tensor_model_parallel_all_reduce_impl(input_)


def tbo_all_reduce_launch(
    input_: torch.Tensor,
    comm_stream: Optional["torch.cuda.Stream"],
) -> Tuple[torch.Tensor, Optional["torch.cuda.Event"]]:
    """Launch a TP all-reduce on ``comm_stream`` (TBO); pair with the wait below.

    Falls back to a synchronous all-reduce (``done_event=None``) when there is
    no comm stream or TP is trivial.
    """
    if comm_stream is None or get_tp_group().world_size == 1:
        return _tensor_model_parallel_all_reduce_impl(input_), None

    comm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(comm_stream):
        output = get_tp_group().all_reduce(input_)
    input_.record_stream(comm_stream)  # keep input_ alive while comm uses it
    done_event = torch.cuda.Event()
    done_event.record(comm_stream)
    return output, done_event


def tbo_all_reduce_wait(
    output: torch.Tensor,
    done_event: Optional["torch.cuda.Event"],
) -> torch.Tensor:
    """Make the compute stream wait for a ``tbo_all_reduce_launch`` to finish."""
    if done_event is not None:
        torch.cuda.current_stream().wait_event(done_event)
    return output


def tensor_model_parallel_quant_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().quant_all_reduce(input_)


def tensor_model_parallel_fused_allreduce_rmsnorm(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fused TP all-reduce + RMSNorm.

    Policy and backend selection are owned by GroupCoordinator:
    it may dispatch to communicator-native fused APIs, custom fused kernels,
    or return None so callers can run generic fallback paths.
    """
    return get_tp_group().fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


def attention_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across attention parallel group."""
    return get_attn_tp_group().all_reduce(input_)


def attention_tensor_model_parallel_quant_all_reduce(
    input_: torch.Tensor,
) -> torch.Tensor:
    """All-reduce the input tensor across attention parallel group."""
    return get_attn_tp_group().quant_all_reduce(input_)


def moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe parallel group."""
    return get_moe_tp_group().all_reduce(input_)


def moe_expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe expert parallel group."""
    return get_moe_ep_group().all_reduce(input_)
