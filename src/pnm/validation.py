"""Independent FP32 reference, shared error definitions and report serialization."""
import json
from pathlib import Path
import torch


def reference_attention(q, k, v, query_start=0, mask=None, chunk_size=64):
    # Explicit per-KV-group calculation: no SDPA and no repeat_kv helper from DUT.
    b, hq, tq, d = q.shape
    hk, tk = k.shape[1:3]
    groups = hq // hk
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for h in range(hk):
        for start in range(0, tq, chunk_size):
            end = min(tq, start + chunk_size)
            scores = q[:, h*groups:(h+1)*groups, start:end].float() @ k[:, h:h+1].float().transpose(-1, -2)
            scores *= d ** -0.5
            row_pos = query_start + torch.arange(start, end, device=q.device)
            allowed = torch.arange(tk, device=q.device)[None, :] <= row_pos[:, None]
            scores.masked_fill_(~allowed, -torch.inf)
            if mask is not None:
                m = mask[..., start:end, :]
                if m.dtype == torch.bool:
                    scores.masked_fill_(~m, -torch.inf)
                else:
                    scores += m.float()
            probs = torch.nan_to_num(scores.softmax(-1), nan=0.)
            out[:, h*groups:(h+1)*groups, start:end] = probs @ v[:, h:h+1].float()
    return out


def errors(actual, reference, relative_floor=1e-3):
    a, b = actual.detach().float(), reference.detach().float()
    diff = (a - b).abs()
    norm=a.norm()*b.norm()
    cosine=float(torch.dot(a.flatten(),b.flatten())/norm) if norm>0 else float(not a.any() and not b.any())
    return {"max_abs": diff.max().item(), "mean_abs": diff.mean().item(),
            "cosine_similarity":cosine,
            "max_relative": (diff / b.abs().clamp_min(relative_floor)).max().item(),
            "rms": diff.square().mean().sqrt().item(), "relative_floor": relative_floor,
            "reference_abs_max": b.abs().max().item(), "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}


def check_close(actual, reference, atol=0.02, rtol=0.02):
    result = errors(actual, reference)
    result.update(atol=atol, rtol=rtol)
    result["pass"] = result["finite"] and bool(torch.allclose(actual.float(), reference.float(), atol=atol, rtol=rtol))
    if not result["pass"]:
        raise AssertionError(json.dumps(result))
    return result


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload=json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    temporary=Path(str(path)+'.tmp')
    temporary.write_text(payload)
    temporary.replace(path)
