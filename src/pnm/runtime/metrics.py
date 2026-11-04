import json
from pathlib import Path


class Metrics:
    """Streaming metrics; do not retain tensors or per-token records in RAM."""
    def __init__(self, path=None):
        self.file = None
        self.counts = {}
        self.bytes = {}
        self.milliseconds = {}
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.file = open(path, "w", buffering=1)

    def emit(self, record):
        stage = record["stage"]
        self.counts[stage] = self.counts.get(stage, 0) + 1
        self.bytes[stage] = self.bytes.get(stage, 0) + record.get("bytes", 0)
        self.milliseconds[stage] = self.milliseconds.get(stage, 0.) + record.get("ms", 0.)
        if self.file:
            self.file.write(json.dumps(record) + "\n")

    def summary(self):
        return {"counts": dict(self.counts), "bytes": dict(self.bytes), "milliseconds": dict(self.milliseconds)}

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
