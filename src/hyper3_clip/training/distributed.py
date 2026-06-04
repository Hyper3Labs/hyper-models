from __future__ import annotations

import os
from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn import all_gather as differentiable_all_gather


def init_distributed() -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(get_local_rank())
        dist.init_process_group(backend=backend)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def destroy_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def gather_with_grad(tensor: Tensor) -> Tensor:
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    return torch.cat(list(differentiable_all_gather(tensor.contiguous())), dim=0)


def gather_variable_with_grad(tensor: Tensor) -> tuple[Tensor, Tensor]:
    """Gather tensors with variable first-dimension lengths across ranks."""
    count_tensor, max_count, keep = _variable_gather_metadata(tensor)
    if get_world_size() == 1:
        return tensor, count_tensor
    return _gather_variable_from_metadata(tensor, max_count, keep), count_tensor


def gather_variable_many_with_grad(tensors: Sequence[Tensor]) -> tuple[list[Tensor], Tensor]:
    """Gather same-length variable tensors while sharing count metadata.

    Tensors with matching dtype/rank/trailing shape are packed along the last
    dimension so a single differentiable all-gather can serve several feature
    tensors with the same variable first dimension.
    """
    if not tensors:
        raise ValueError("gather_variable_many_with_grad requires at least one tensor")
    first = tensors[0]
    for tensor in tensors:
        if tensor.device != first.device:
            raise ValueError("all tensors must be on the same device")
        if tensor.shape[0] != first.shape[0]:
            raise ValueError("all tensors must have the same first dimension")
    count_tensor, max_count, keep = _variable_gather_metadata(first)
    if get_world_size() == 1:
        return list(tensors), count_tensor

    gathered: list[Tensor | None] = [None] * len(tensors)
    groups: dict[tuple[torch.dtype, torch.Size, int], list[int]] = {}
    for index, tensor in enumerate(tensors):
        if tensor.dim() == 0:
            raise ValueError("variable gather tensors must have at least one dimension")
        key = (
            (tensor.dtype, tensor.shape[1:-1], tensor.dim())
            if tensor.dim() > 1
            else (tensor.dtype, torch.Size(), 1)
        )
        groups.setdefault(key, []).append(index)

    for indices in groups.values():
        group_tensors = [tensors[index] for index in indices]
        if len(group_tensors) == 1 or group_tensors[0].dim() == 1:
            for index, tensor in zip(indices, group_tensors, strict=True):
                gathered[index] = _gather_variable_from_metadata(tensor, max_count, keep)
            continue
        widths = [tensor.shape[-1] for tensor in group_tensors]
        packed = torch.cat(group_tensors, dim=-1)
        gathered_packed = _gather_variable_from_metadata(packed, max_count, keep)
        for index, chunk in zip(indices, gathered_packed.split(widths, dim=-1), strict=True):
            gathered[index] = chunk

    if any(tensor is None for tensor in gathered):
        raise RuntimeError("internal error while gathering variable tensors")
    return [tensor for tensor in gathered if tensor is not None], count_tensor


def gather_variable_no_grad(tensor: Tensor) -> tuple[Tensor, Tensor]:
    """Gather variable-length tensors that do not require autograd."""
    count_tensor, max_count, keep = _variable_gather_metadata(tensor)
    if get_world_size() == 1:
        return tensor, count_tensor
    padded = tensor.new_zeros((max_count, *tensor.shape[1:]))
    padded[: tensor.shape[0]] = tensor
    gathered = [torch.zeros_like(padded) for _ in range(get_world_size())]
    dist.all_gather(gathered, padded.contiguous())
    return torch.cat(gathered, dim=0)[keep], count_tensor


def _variable_gather_metadata(tensor: Tensor) -> tuple[Tensor, int, Tensor]:
    world_size = get_world_size()
    local_count = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    if world_size == 1:
        keep = torch.ones(tensor.shape[0], device=tensor.device, dtype=torch.bool)
        return local_count, tensor.shape[0], keep

    counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(counts, local_count)
    count_tensor = torch.cat(counts)
    max_count = int(count_tensor.max().item())
    keep = torch.zeros(world_size * max_count, device=tensor.device, dtype=torch.bool)
    for rank, count in enumerate(count_tensor.tolist()):
        start = rank * max_count
        keep[start : start + count] = True
    return count_tensor, max_count, keep


def _gather_variable_from_metadata(tensor: Tensor, max_count: int, keep: Tensor) -> Tensor:
    padded_shape = (max_count, *tensor.shape[1:])
    padded = tensor.new_zeros(padded_shape)
    padded[: tensor.shape[0]] = tensor

    gathered = torch.cat(list(differentiable_all_gather(padded.contiguous())), dim=0)
    return gathered[keep]


def local_target_indices(batch_size: int, device: torch.device) -> Tensor:
    return torch.arange(batch_size, device=device) + batch_size * get_rank()
