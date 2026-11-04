from dataclasses import dataclass
from enum import Enum
from typing import Any


class Op(str, Enum):
    ALLOC = "alloc_cache"
    INIT = "init_cache"
    APPEND = "append_and_attend"
    INFO = "get_cache_info"
    FREE = "free_cache"


class PNMError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class CacheSpec:
    capacity: int
    batch: int = 1
    query_heads: int = 24
    kv_heads: int = 4
    head_dim: int = 256
    dtype: str = "bfloat16"
    layout: str = "BHTD"

    def validate(self):
        if min(self.query_heads, self.kv_heads, self.head_dim) < 1:
            raise PNMError("INVALID_SPEC", "nonpositive dimensions")
        if self.capacity < 1 or self.batch != 1 or self.query_heads % self.kv_heads:
            raise PNMError("INVALID_SPEC", str(self))
        if self.dtype != "bfloat16" or self.layout != "BHTD":
            raise PNMError("INVALID_SPEC", "v1 requires BF16 BHTD")


@dataclass(frozen=True)
class Command:
    command_id: str
    op: Op
    request_id: str
    layer_id: int
    handle: str | None = None
    expected_length: int = 0
    token_position: int = 0
    spec: CacheSpec | None = None
    dtype: str = "bfloat16"
    layout: str = "BHTD"


@dataclass(frozen=True)
class CacheInfo:
    handle: str
    request_id: str
    layer_id: int
    sequence_length: int
    capacity: int
    dtype: str
    layout: str
    allocated_bytes: int
    initialized: bool


@dataclass
class Completion:
    command_id: str
    status: str
    info: CacheInfo | None = None
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None
