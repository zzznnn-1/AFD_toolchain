"""Device-side attention. Positions are explicit, including rectangular causal masks."""
import torch
from torch.nn.functional import scaled_dot_product_attention
from pnm.protocol import PNMError


def attend(q, k, v, query_start=0, mask=None):
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise PNMError("SHAPE", "Q/K/V must be BHTD; K and V must match")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]:
        raise PNMError("SHAPE", "invalid GQA dimensions")
    tq, tk = q.shape[2], k.shape[2]
    if query_start < 0 or query_start + tq > tk:
        raise PNMError("POSITION", "query positions outside valid KV")
    if mask is None and tq == tk and query_start == 0:
        return scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    if mask is None and tq == 1 and query_start == tk - 1:
        # Single decode token sees ALL history, not the upper-left causal triangle.
        return scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
    allowed = torch.arange(tk, device=q.device)[None, :] <= (
        query_start + torch.arange(tq, device=q.device)[:, None])
    if mask is None:
        effective = allowed
    elif mask.dtype == torch.bool:
        effective = allowed & mask
    else:
        effective = mask.masked_fill(~allowed, -torch.inf)
        # This torch/cuDNN build requires additive bias to match Q's dtype.
        effective = effective.to(q.dtype)
    return scaled_dot_product_attention(q, k, v, attn_mask=effective, is_causal=False, enable_gqa=True)
