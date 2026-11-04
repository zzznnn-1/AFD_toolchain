"""Explicit GPU -> pinned RAM -> GPU transport, with a bounded reusable slot pool.

The v1 API completes both copy events before returning. Copies are issued async on
dedicated streams, but commands are serialized; there is no claim of overlap.
"""
import torch
from pathlib import Path
from pnm.protocol import PNMError


class HostStaging:
    def __init__(self, source, target, max_tokens, spec, metrics, chunk_tokens=256):
        self.source, self.target = torch.device(source), torch.device(target)
        if self.source == self.target or self.source.type != "cuda" or self.target.type != "cuda":
            raise ValueError("two distinct CUDA devices required")
        self.metrics = metrics
        self.chunk_tokens = min(max_tokens,chunk_tokens)
        self.transfers = 0
        self.streams = {d: torch.cuda.Stream(device=d) for d in (self.source, self.target)}
        self.slots = {}
        sizes = {"q": spec.query_heads * spec.head_dim,
                 "k": self.chunk_tokens * spec.kv_heads * spec.head_dim,
                 "v": self.chunk_tokens * spec.kv_heads * spec.head_dim,
                 "o": spec.query_heads * spec.head_dim, "mask": max_tokens}
        for name, size in sizes.items():
            dtype = torch.float32 if name == "mask" else torch.bfloat16
            dest = self.source if name == "o" else self.target
            self.slots[name] = (torch.empty(size, dtype=dtype, pin_memory=True),
                                torch.empty(size, dtype=dtype, device=dest))
        self.allocated_bytes = sum(p.numel() * p.element_size() for p, _ in self.slots.values())
        self.all_pinned = all(p.is_pinned() for p, _ in self.slots.values())

    def transfer(self, tensor, name, command_id):
        host, output = self.slots[name]
        src = self.target if name == "o" else self.source
        dst = output.device
        if tensor.device != src or tensor.dtype != host.dtype or tensor.numel() > host.numel():
            raise PNMError("TRANSFER_CONTRACT", f"slot {name}: {tensor.shape} {tensor.dtype} {tensor.device}")
        inp = tensor.contiguous()
        h = host[:inp.numel()].view(inp.shape)
        out = output[:inp.numel()].view(inp.shape)
        s, d = self.streams[src], self.streams[dst]
        s.wait_stream(torch.cuda.current_stream(src))
        with torch.cuda.device(src), torch.cuda.stream(s):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(s)
            h.copy_(inp, non_blocking=True)
            inp.record_stream(s)
            end.record(s)
        end.synchronize()  # Host storage is now readable by destination DMA.
        self.metrics.emit({"command_id": command_id, "stage": f"{src}_d2h", "tensor": name,
                           "bytes": inp.numel() * inp.element_size(), "ms": start.elapsed_time(end), "pinned": h.is_pinned()})
        with torch.cuda.device(dst), torch.cuda.stream(d):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(d)
            out.copy_(h, non_blocking=True)
            end.record(d)
        torch.cuda.current_stream(dst).wait_event(end)
        end.synchronize()  # Both sides finished before reusing this pinned slot.
        self.transfers += 1
        self.metrics.emit({"command_id": command_id, "stage": f"{dst}_h2d", "tensor": name,
                           "bytes": inp.numel() * inp.element_size(), "ms": start.elapsed_time(end), "pinned": h.is_pinned()})
        return out

    def describe(self):
        numa={}
        try:
            ranges=[]
            for line in Path('/proc/self/maps').read_text().splitlines():
                start,end=(int(v,16) for v in line.split()[0].split('-'))
                ranges.append((start,end))
            maps={int(line.split()[0],16):line for line in Path('/proc/self/numa_maps').read_text().splitlines()}
            for name,(host,_) in self.slots.items():
                region=next((start for start,end in ranges if start<=host.data_ptr()<end),None)
                numa[name]={'vma':maps.get(region),'scope':'whole containing VMA, not exact buffer page placement'}
        except OSError as exc:
            numa={'unavailable':str(exc)}
        return {'transport':'host_staging','pinned_bytes':self.allocated_bytes,'all_pinned':self.all_pinned,
                'chunk_tokens':self.chunk_tokens,'preallocated_slots':len(self.slots),
                'transfer_calls':self.transfers,'pool_resizes':0,'host_numa_vmas':numa}

    def close(self):
        for s in self.streams.values():
            s.synchronize()
        self.slots.clear()
        self.streams.clear()
