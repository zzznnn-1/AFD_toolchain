"""Local-device CUDA spans and CPU wall clock; never compare cross-GPU timestamps."""
from time import perf_counter
import torch


class Span:
    def __init__(self, device):
        self.device=torch.device(device)
        self.wall_start=perf_counter()
        self.start=self.end=None
        if self.device.type=='cuda':
            with torch.cuda.device(self.device):
                self.start=torch.cuda.Event(enable_timing=True)
                self.end=torch.cuda.Event(enable_timing=True)
                self.start.record()

    def finish(self):
        cuda_ms=None
        if self.start is not None:
            with torch.cuda.device(self.device): self.end.record()
            self.end.synchronize()
            cuda_ms=self.start.elapsed_time(self.end)
        return {'cuda_ms':cuda_ms,'wall_ms':(perf_counter()-self.wall_start)*1000}
