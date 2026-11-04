"""Versioned simulator command envelope. This is NOT an ASIC DMA/ioctl ABI."""
from dataclasses import asdict
import json
import struct
import zlib
from .types import PNMError


HEADER = struct.Struct('<4sHII')


def encode(command, payload):
    tensors = {}
    for name,tensor in payload.items():
        if tensor is None:
            continue
        tensors[name] = {'shape':list(tensor.shape),'stride':list(tensor.stride()),
                         'dtype':str(tensor.dtype),'device':str(tensor.device),
                         'bytes':tensor.numel()*tensor.element_size()}
    data = {'command':asdict(command),'tensors':tensors}
    body = json.dumps(data,sort_keys=True,separators=(',',':')).encode()
    return HEADER.pack(b'PNM1',1,len(body),zlib.crc32(body))+body


def decode(packet):
    if len(packet)<HEADER.size:
        raise PNMError('PROTOCOL_FRAME','truncated header')
    magic,version,size,checksum=HEADER.unpack_from(packet)
    body=packet[HEADER.size:]
    if magic!=b'PNM1' or version!=1 or size!=len(body) or size>1024*1024:
        raise PNMError('PROTOCOL_FRAME','unsupported version/size/magic')
    if zlib.crc32(body)!=checksum:
        raise PNMError('PROTOCOL_CHECKSUM','command envelope changed')
    data=json.loads(body)
    if data['command']['version']!=1:
        raise PNMError('PROTOCOL_VERSION','unsupported command version')
    return data
