from torch.distributed import ReduceOp

MB = 1024 * 1024


def all_reduce_coalesced(tensors, divisor=1, op=ReduceOp.SUM, buffer_size=256 * MB):
    raise NotImplementedError("all_reduce_coalesced was removed from the codebase.")


def all_gather_coalesced(tensors, buffer_size=256 * MB):
    raise NotImplementedError("all_gather_coalesced was removed from the codebase.")


def broadcast_coalesced(tensors, src=0, buffer_size=10 * MB):
    raise NotImplementedError("broadcast_coalesced was removed from the codebase.")
